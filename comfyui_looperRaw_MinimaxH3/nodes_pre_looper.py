import os
import time
import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

try:
    import folder_paths
except ImportError:
    folder_paths = None

from .looper_utils import (
    get_video_path,
    get_video_metadata,
    load_video_frames,
    scale_image_to_total_pixels,
    extract_video_audio,
    slice_audio,
    snap_to_minimax_generation_length,
    minimax_frames_to_temporal,
    create_latent_dict
)
from .minimax_core import (
    apply_minimax_h3_guide,
    call_minimax_h3_reference,
    patch_minimax_h3_continuity
)

class PreLooperMinimaxNode:
    """
    Pre Looper node for MiniMax H3.
    Directly selects and uploads videos (.mp4, .mov, etc.), provides interactive timeline controls,
    scales video to megapixels (0.5 default Lanczos, multiples of 32),
    and executes MiniMaxH3ReferenceToVideo + MiniMaxH3AddGuide internally to eliminate circular node connections.
    """
    @classmethod
    def INPUT_TYPES(cls):
        video_files = []
        try:
            if folder_paths is not None:
                if hasattr(folder_paths, "get_filename_list"):
                    try:
                        video_files = list(folder_paths.get_filename_list("video") or [])
                    except Exception:
                        pass
                if not video_files:
                    input_dir = folder_paths.get_input_directory()
                    if os.path.isdir(input_dir):
                        video_exts = ('.mp4', '.mov', '.avi', '.webm', '.mkv', '.gif', '.flv', '.wmv')
                        for root, dirs, files in os.walk(input_dir):
                            dirs[:] = [d for d in dirs if not d.startswith('.')]
                            for f in files:
                                if f.lower().endswith(video_exts):
                                    rel_path = os.path.relpath(os.path.join(root, f), input_dir)
                                    video_files.append(rel_path.replace("\\", "/"))
        except Exception:
            pass
        video_files.sort()
        if not video_files:
            video_files = ["none.mp4"]

        return {
            "required": {
                "video": (video_files,),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "megapixels": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 4.0, "step": 0.05}),
                "trans_duration": ("INT", {"default": 50, "min": 5, "max": 1000, "step": 1}),
                "head_a": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "tail_b": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "seam_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Frame where new video begins (cut between Chunk A and B). 0 = auto middle split"}),
            },
            "optional": {
                "load_video": ("IMAGE", {"tooltip": "Frames de video opcionales (IMAGE tensor) desde un nodo Load Video externo (ej. VHS_LoadVideo). Tiene máxima prioridad sobre el selector/botón de video."}),
                "load_audio": ("AUDIO", {"tooltip": "Audio opcional (AUDIO) desde un nodo Load Video externo. Tiene prioridad sobre el audio del archivo de video."}),
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "Metadatos opcionales desde VHS_LoadVideo (para FPS y duración precisa)."}),
                "audio_vae": ("VAE",),
                "model": ("MODEL", {"tooltip": "Optional MiniMax H3 Diffusion Model. Connect here to apply continuity patching / circular wrapping directly inside Pre Looper."}),
                "prompt_assist": ("BOOLEAN", {"default": True, "tooltip": "Format prompt automatically with MiniMax H3 <Picture 1> anchor, continuous motion directive, and overall_soundscape"}),
                "guide_frames": ("INT", {"default": 22, "min": 5, "max": 39, "step": 1, "tooltip": "Context guide frames per boundary (22 = 7 tokens standard, 17 = 5 tokens for greater prompt freedom and faster sampling)"}),
                "continuity_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Integrated continuity patcher strength on `model` (0.0 = passthrough, 0.25-0.40 = active boundary lock)"}),
                "pad_clips": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1, "tooltip": "17-frame clips of circular wrapped context added on each side during diffusion (0 = disabled, 1 = circular wrap with fresh noise swap)"}),
                "blend_tokens": ("INT", {"default": 2, "min": 0, "max": 7, "step": 1, "tooltip": "Cosine ramp blend tokens inside the context boundary when continuity_strength > 0"}),
                "sigma_schedule": ("BOOLEAN", {"default": True, "tooltip": "Flow-Matching sigma schedule: anchors trajectory at high sigma and smoothly releases as sigma -> 0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "CONDITIONING", "LATENT", "AUDIO", "AUDIO", "MODEL")
    RETURN_NAMES = ("chunk_a", "chunk_b", "positive", "latent", "chunk_audio_a", "chunk_audio_b", "model")
    FUNCTION = "process"
    CATEGORY = "LooperMinimaxH3"

    def process(self, video: str, clip, vae, prompt: str, megapixels: float,
                trans_duration: int, head_a: int = 0, tail_b: int = 0,
                seam_frame: int = 0, audio_vae=None, model=None,
                load_video=None, load_audio=None, video_info=None,
                prompt_assist: bool = True, guide_frames: int = 22,
                continuity_strength: float = 0.0, pad_clips: int = 0,
                blend_tokens: int = 2, sigma_schedule: bool = True, **kwargs):
        
        # Support input slot aliases
        if load_video is None:
            load_video = kwargs.get("video_images") or kwargs.get("images") or kwargs.get("video_frames")
        if load_audio is None:
            load_audio = kwargs.get("video_audio") or kwargs.get("audio")
        if video_info is None:
            video_info = kwargs.get("info")

        t_start = time.perf_counter()
        pbar = ProgressBar(100) if ProgressBar is not None else None
        if pbar is not None:
            pbar.update_absolute(5, 100)

        is_external_video = load_video is not None
        source_label = "Load Video Link (IMAGE tensor)" if is_external_video else f"'{video}'"

        print("\n" + "=" * 65)
        print(f"[LooperMinimaxH3] >>> PREPROCESS START: Source: {source_label}")
        print("=" * 65)

        # 1. Load video frames (External Load Video node takes PRIORITY over widget file)
        if is_external_video:
            print(f"[LooperMinimaxH3] >>> PRIORIDAD ACTIVA: Utilizando video enlazado desde nodo Load Video externo.")
            if not isinstance(load_video, torch.Tensor):
                raise TypeError(f"[LooperMinimaxH3] load_video debe ser un torch.Tensor (IMAGE), recibido: {type(load_video)}")
            raw_frames = load_video
            if raw_frames.ndim == 3:
                raw_frames = raw_frames.unsqueeze(0)
            elif raw_frames.ndim == 4 and raw_frames.shape[1] in (1, 3, 4) and raw_frames.shape[-1] not in (1, 3, 4):
                raw_frames = raw_frames.permute(0, 2, 3, 1)

            elapsed_load = 0.0
            total_frames = raw_frames.shape[0]

            # Resolving FPS: video_info > local file metadata > fallback 24.0
            fps = 24.0
            if video_info and isinstance(video_info, dict) and "fps" in video_info:
                try:
                    fps = float(video_info["fps"] or 24.0)
                except (ValueError, TypeError):
                    fps = 24.0
            elif video and video != "none.mp4":
                try:
                    vpath = get_video_path(video)
                    if os.path.isfile(vpath):
                        meta = get_video_metadata(vpath)
                        fps = float(meta.get("fps", 24.0) or 24.0)
                except Exception:
                    pass

            # Resolving Audio: load_audio > local file audio > None
            video_audio = None
            if load_audio is not None and isinstance(load_audio, dict) and "waveform" in load_audio and load_audio["waveform"].shape[-1] > 0:
                video_audio = load_audio
                print(f"[LooperMinimaxH3] Utilizando audio desde enlace de nodo Load Video.")
            elif video and video != "none.mp4":
                try:
                    vpath = get_video_path(video)
                    if os.path.isfile(vpath):
                        video_audio = extract_video_audio(vpath)
                except Exception:
                    video_audio = None
        else:
            video_path = get_video_path(video)
            print(f"[LooperMinimaxH3] Loading video file: {video_path}")
            t_load = time.perf_counter()
            raw_frames = load_video_frames(video_path, pbar=pbar, pbar_range=(5, 45))
            elapsed_load = time.perf_counter() - t_load
            total_frames = raw_frames.shape[0]

            video_audio = extract_video_audio(video_path)
            meta = get_video_metadata(video_path)
            fps = float(meta.get("fps", 24.0) or 24.0)

        if pbar is not None:
            pbar.update_absolute(45, 100)

        # 2. Validate frame count
        head_a = max(0, head_a)
        tail_b = max(0, tail_b)

        # Smart disambiguation: detect if tail_b was passed as absolute frame index
        if tail_b > (total_frames // 2) and tail_b > head_a and (head_a + tail_b + 44 > total_frames) and tail_b <= total_frames:
            converted_tail = max(0, total_frames - tail_b)
            print(f"[LooperMinimaxH3] Interpreting tail_b={tail_b} as absolute end frame -> cut from end: {converted_tail}f")
            tail_b = converted_tail

        # Enforce minimum 44 frames active window (22 frames for Chunk A + 22 frames for Chunk B)
        max_possible_tail = max(0, total_frames - head_a - 44)
        if tail_b > max_possible_tail and total_frames >= 44:
            print(f"[LooperMinimaxH3] Clamping tail_b from {tail_b}f to {max_possible_tail}f to preserve 44f minimum context.")
            tail_b = max_possible_tail

        min_required = head_a + tail_b + 44
        if total_frames < min_required:
            raise ValueError(
                f"[LooperMinimaxH3] Video has {total_frames} frames, but at least {min_required} frames are required "
                f"for head_a ({head_a}) + tail_b ({tail_b}) and 44 frames minimum context for MiniMax H3. "
                f"Adjust head_a or tail_b."
            )

        # 3. First step: Scale to total megapixels using Lanczos, divisible by 32
        t_scale = time.perf_counter()
        frames, target_w, target_h = scale_image_to_total_pixels(
            raw_frames,
            megapixels=megapixels,
            upscale_method="lanczos",
            divisible_by=32,
            pbar=pbar,
            pbar_step=70
        )
        elapsed_scale = time.perf_counter() - t_scale
        if pbar is not None:
            pbar.update_absolute(70, 100)

        # 4. Trim head and tail (Active body segment)
        end_idx = total_frames - tail_b
        active_segment = frames[head_a:end_idx]
        active_length = active_segment.shape[0]

        # 5. Split active segment into chunk_a and chunk_b
        # Lock 22 frames minimum for Chunk A and 22 frames minimum for Chunk B
        min_seam = head_a + 22
        max_seam = end_idx - 22

        if min_seam > max_seam:
            raise ValueError(
                f"[LooperMinimaxH3] Active segment has {active_length} frames, but at least 44 frames "
                f"are required (22 for Chunk A and 22 for Chunk B). Adjust head_a or tail_b."
            )

        if seam_frame <= 0:
            # Default: auto middle split
            half_point = active_length // 2
            actual_seam_frame = head_a + half_point
            actual_seam_frame = max(min_seam, min(actual_seam_frame, max_seam))
        else:
            # Clamped user-selected seam frame (new video start)
            actual_seam_frame = max(min_seam, min(int(seam_frame), max_seam))

        split_point = actual_seam_frame - head_a
        chunk_a = active_segment[:split_point]
        chunk_b = active_segment[split_point:]
        seam_frame = actual_seam_frame

        if chunk_a.shape[0] < 22 or chunk_b.shape[0] < 22:
            raise ValueError(
                f"[LooperMinimaxH3] Each chunk must have at least 22 frames for MiniMax H3 guidance. "
                f"Currently chunk_a has {chunk_a.shape[0]} and chunk_b has {chunk_b.shape[0]}."
            )

        if pbar is not None:
            pbar.update_absolute(75, 100)

        # 6. Extract audio slices and guide frames matching video operations 1:1
        t_audio = time.perf_counter()

        chunk_audio_a = None
        chunk_audio_b = None
        guide_a_audio = None
        guide_b_audio = None

        if video_audio is not None and "waveform" in video_audio and video_audio["waveform"].shape[-1] > 0:
            sr = video_audio["sample_rate"]
            waveform = video_audio["waveform"]
            total_samples = waveform.shape[-1]

            # 1:1 mapping from video frame indexes to exact audio sample positions:
            # - start: head_a cut point
            # - seam: exact middle cut point (seam_frame)
            # - end: tail_b cut point (end_idx)
            start_samp = max(0, int(round(head_a * sr / fps)))
            seam_samp = min(total_samples, int(round(seam_frame * sr / fps)))
            end_samp = min(total_samples, int(round(end_idx * sr / fps)))

            chunk_audio_a = {
                "waveform": waveform[..., start_samp:seam_samp],
                "sample_rate": sr
            }
            chunk_audio_b = {
                "waveform": waveform[..., seam_samp:end_samp],
                "sample_rate": sr
            }

            eff_guide_f = max(5, min(int(guide_frames), chunk_a.shape[0], chunk_b.shape[0]))
            guide_a_end_samp = min(total_samples, int(round((head_a + eff_guide_f) * sr / fps)))
            guide_b_start_samp = max(0, int(round((end_idx - eff_guide_f) * sr / fps)))

            guide_a_audio = {
                "waveform": waveform[..., start_samp:guide_a_end_samp],
                "sample_rate": sr
            }
            guide_b_audio = {
                "waveform": waveform[..., guide_b_start_samp:end_samp],
                "sample_rate": sr
            }

        empty_audio = {"waveform": torch.zeros((1, 2, 0), dtype=torch.float32), "sample_rate": 44100}
        out_audio_a = chunk_audio_a if chunk_audio_a is not None else empty_audio
        out_audio_b = chunk_audio_b if chunk_audio_b is not None else empty_audio
        elapsed_audio = time.perf_counter() - t_audio

        eff_guide_f = max(5, min(int(guide_frames), chunk_a.shape[0], chunk_b.shape[0]))
        guide_frames_a = chunk_a[:eff_guide_f]
        guide_frames_b = chunk_b[-eff_guide_f:]

        # Reference images for Qwen vision tokenizer:
        # The transition originates at the end of Chunk B and arrives at the start of Chunk A:
        # image_0 (<Picture 1>): last frame of Chunk B context (origin of transition, anchors frame_idx = 0)
        image_0 = guide_frames_b[-1:]

        # image_1 (<Picture 2>): first frame of Chunk A context (destination of transition)
        image_1 = guide_frames_a[0:1]

        # 7. Total generation length snapped to native MiniMax H3 17k + 5 boundary
        duration_length = snap_to_minimax_generation_length(
            trans_duration + 2 * eff_guide_f,
            min_frames=2 * eff_guide_f + 5
        )

        # 8. Directly execute MiniMaxH3ReferenceToVideo to generate base conditioning & latent
        if pbar is not None:
            pbar.update_absolute(80, 100)

        ref_audios = None
        if audio_vae is not None and guide_b_audio is not None and guide_b_audio["waveform"].shape[-1] > 0:
            ref_audios = {"ref_audio_0": guide_b_audio, "ref_audio_1": guide_a_audio}

        t_ref = time.perf_counter()
        positive, latent = call_minimax_h3_reference(
            clip=clip,
            vae=vae,
            image_0=image_0,
            image_1=image_1,
            width=target_w,
            height=target_h,
            length=duration_length,
            prompt=prompt,
            audio_vae=audio_vae,
            ref_audios=ref_audios,
            auto_format_prompt=prompt_assist
        )
        elapsed_ref = time.perf_counter() - t_ref

        if positive is None or latent is None:
            raise RuntimeError(
                "[LooperMinimaxH3] Failed to generate conditioning and latent using MiniMaxH3ReferenceToVideo. "
                "Ensure MiniMax H3 models and comfy_extras are properly installed."
            )

        if isinstance(latent, dict):
            latent["guide_overlap"] = eff_guide_f

        if pbar is not None:
            pbar.update_absolute(88, 100)

        # 9. Inject MiniMax H3 Guide frames & audio into the positive conditioning
        positive = apply_minimax_h3_guide(
            positive=positive,
            latent=latent,
            vae=vae,
            image=guide_frames_a,
            frame_idx=-eff_guide_f,
            audio=guide_a_audio if audio_vae is not None else None,
            audio_vae=audio_vae
        )
        positive = apply_minimax_h3_guide(
            positive=positive,
            latent=latent,
            vae=vae,
            image=guide_frames_b,
            frame_idx=0,
            audio=guide_b_audio if audio_vae is not None else None,
            audio_vae=audio_vae
        )

        # 10. Integrated Continuity Patcher on optional `model` input
        patched_model = None
        if model is not None:
            k_tok = minimax_frames_to_temporal(eff_guide_f)
            guide_lat_b = None
            guide_lat_a = None
            if continuity_strength > 0.0:
                try:
                    if isinstance(positive, list) and len(positive) > 0 and isinstance(positive[0], (tuple, list)) and len(positive[0]) > 1:
                        kfs = positive[0][1].get("minimax_keyframes", [])
                        if len(kfs) >= 2 and "latent" in kfs[0] and "latent" in kfs[1]:
                            guide_lat_a = create_latent_dict(kfs[0]["latent"], kfs[0].get("audio_latent"))
                            guide_lat_b = create_latent_dict(kfs[1]["latent"], kfs[1].get("audio_latent"))
                    if guide_lat_b is None or guide_lat_a is None:
                        with torch.no_grad():
                            enc_b = vae.encode(guide_frames_b)
                            enc_a = vae.encode(guide_frames_a)
                        guide_lat_b = create_latent_dict(enc_b)
                        guide_lat_a = create_latent_dict(enc_a)
                except Exception as e:
                    print(f"[LooperMinimaxH3] Warning encoding guide latents for model patch: {e}")

            patched_model = patch_minimax_h3_continuity(
                model=model,
                chunk_latent_b=guide_lat_b,
                chunk_latent_a=guide_lat_a,
                k_tokens=k_tok,
                blend_tokens=blend_tokens,
                strength=continuity_strength,
                pad_clips=pad_clips,
                sigma_schedule=sigma_schedule
            )

        if pbar is not None:
            pbar.update_absolute(100, 100)

        t_total = time.perf_counter() - t_start

        # Detailed VHS-style Preprocess Metrics Summary in console
        audio_status = f"Extracted ({out_audio_a['waveform'].shape[-1]} samples)" if (video_audio is not None and video_audio["waveform"].shape[-1] > 0) else "No audio track"
        print("-" * 65)
        print(f"[LooperMinimaxH3] >>> PREPROCESS METRICS SUMMARY:")
        print(f"[LooperMinimaxH3]  - Video Source: {source_label} ({total_frames} frames in {elapsed_load:.2f}s)")
        print(f"[LooperMinimaxH3]  - Resizing/Scale: {target_w}x{target_h} ({megapixels} MP Lanczos) in {elapsed_scale:.2f}s")
        print(f"[LooperMinimaxH3]  - Audio Track: {audio_status} in {elapsed_audio:.2f}s")
        print(f"[LooperMinimaxH3]  - Active Block: {active_length}f (Head Cut: {head_a}f | Tail Cut: {tail_b}f)")
        print(f"[LooperMinimaxH3]  - Chunk A: {chunk_a.shape[0]}f (Frames {head_a} .. {seam_frame - 1})")
        print(f"[LooperMinimaxH3]  - Middle Seam: Frame {seam_frame} (Exact Cut Junction)")
        print(f"[LooperMinimaxH3]  - Chunk B: {chunk_b.shape[0]}f (Frames {seam_frame} .. {end_idx - 1})")
        print(f"[LooperMinimaxH3]  - MiniMax H3 Reference & Guides: {duration_length}f in {elapsed_ref:.2f}s")
        if model is not None:
            print(f"[LooperMinimaxH3]  - Integrated Model Patch: strength={continuity_strength:.2f}, pad_clips={pad_clips}, sigma_schedule={sigma_schedule}")
        print(f"[LooperMinimaxH3]  - TOTAL PREPROCESS TIME: {t_total:.2f}s")
        print("=" * 65 + "\n")

        return (
            chunk_a,
            chunk_b,
            positive,
            latent,
            out_audio_a,
            out_audio_b,
            patched_model
        )
