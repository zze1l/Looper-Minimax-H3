import os
import time
import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None

from .looper_utils import (
    minimax_temporal_to_frames,
    minimax_frames_to_temporal,
    snap_to_minimax_token_boundary,
    snap_frames_to_minimax,
    snap_to_minimax_generation_length,
    extract_video_audio_latents,
    slice_minimax_latent,
    concat_minimax_latents,
    create_latent_dict,
    MINIMAX_FRAME_RESCALE,
    save_preview_video
)


from .minimax_core import (
    apply_minimax_h3_guide,
    apply_minimax_h3_continuation_guides,
    call_minimax_h3_reference,
    patch_minimax_h3_continuity
)

class PreLooperLatentMinimaxNode:
    """
    Pre Looper Latent node for MiniMax H3 (PreLooperLatentMinimax).
    Homologous to PreLooperMinimaxNode, operating 100% in latent space:
    - Receives incoming video latent (NestedTensor or standard 5D tensor)
    - Performs head/tail cuts and middle seam splitting natively in latent space
    - Extracts 22-frame context guides (7 latent tokens)
    - Generates base conditioning and empty transition latent for SamplerCustomAdvanced
    - TIP: Wire chunk_latent_b and chunk_latent_a to LooperContinuityPatcher to eliminate
      diffusion boundary drift and guarantee 100% seamless transition continuity!
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Incoming video latent to loop (e.g. from EmptyMiniMaxH3LatentAV or sampler output)"}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "trans_duration": ("INT", {"default": 50, "min": 5, "max": 1000, "step": 1, "tooltip": "Net duration frames for the transition"}),
                "head_a": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Frames to cut from the start of Chunk A"}),
                "tail_b": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Frames to cut from the end of Chunk B"}),
                "seam_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Frame where new video begins (cut between Chunk A and B). 0 = auto middle split"}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "model": ("MODEL", {"tooltip": "Optional MiniMax H3 Diffusion Model. Connect here to apply continuity patching / circular wrapping directly inside Pre Looper."}),
                "prompt_assist": ("BOOLEAN", {"default": True, "tooltip": "Format prompt automatically with MiniMax H3 <Picture 1> anchor, continuous motion directive, and overall_soundscape"}),
                "guide_tokens": ("INT", {"default": 7, "min": 2, "max": 12, "step": 1, "tooltip": "Latent context guide tokens per boundary (7 tokens = 22f standard, 5 tokens = 17f for higher prompt freedom and faster sampling)"}),
                "continuity_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Integrated continuity patcher strength on `model` (0.0 = passthrough, 0.25-0.40 = active boundary lock)"}),
                "pad_clips": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1, "tooltip": "17-frame clips of circular wrapped context added on each side during diffusion (0 = disabled, 1 = circular wrap with fresh noise swap)"}),
                "blend_tokens": ("INT", {"default": 2, "min": 0, "max": 7, "step": 1, "tooltip": "Cosine ramp blend tokens inside the context boundary when continuity_strength > 0"}),
                "sigma_schedule": ("BOOLEAN", {"default": True, "tooltip": "Flow-Matching sigma schedule: anchors trajectory at high sigma and smoothly releases as sigma -> 0"}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "CONDITIONING", "LATENT", "AUDIO", "AUDIO", "MODEL")
    RETURN_NAMES = ("chunk_latent_a", "chunk_latent_b", "positive", "latent", "chunk_audio_a", "chunk_audio_b", "model")
    FUNCTION = "process"
    CATEGORY = "LooperMinimaxH3"

    def process(self, latent: dict, clip, vae, prompt: str,
                trans_duration: int, head_a: int = 0, tail_b: int = 0,
                seam_frame: int = 0, audio_vae=None, model=None,
                prompt_assist: bool = True, guide_tokens: int = 7,
                continuity_strength: float = 0.0, pad_clips: int = 0,
                blend_tokens: int = 2, sigma_schedule: bool = True, **kwargs):
        
        # Override cuts from optional connected inputs if provided
        opt_h = kwargs.get("opt_head_a", -1)
        if opt_h is not None and opt_h >= 0:
            head_a = opt_h
        opt_t = kwargs.get("opt_tail_b", -1)
        if opt_t is not None and opt_t >= 0:
            tail_b = opt_t
        opt_s = kwargs.get("opt_seam_frame", -1)
        if opt_s is not None and opt_s >= 0:
            seam_frame = opt_s
        
        t_start = time.perf_counter()
        pbar = ProgressBar(100) if ProgressBar is not None else None
        if pbar is not None:
            pbar.update_absolute(5, 100)

        print("\n" + "=" * 65)
        print(f"[LooperMinimaxH3] >>> PREPROCESS LATENT START")
        print("=" * 65)

        # 1. Inspect incoming latent tensor
        video_tensor, audio_tensor = extract_video_audio_latents(latent)
        if video_tensor is None or video_tensor.ndim != 5:
            raise ValueError(f"[LooperMinimaxH3] Expected a 5D video latent tensor [B, C, T, H, W], got {video_tensor.shape if video_tensor is not None else None}")

        total_tokens = video_tensor.shape[2]
        target_h = video_tensor.shape[3] * 16
        target_w = video_tensor.shape[4] * 16
        total_frames = minimax_temporal_to_frames(total_tokens)

        # 2. Convert cuts to latent token indices
        head_a = max(0, head_a)
        tail_b = max(0, tail_b)

        # Smart disambiguation: detect if tail_b was passed as absolute frame index (e.g. 73 in a 107f video)
        # instead of relative cut frames from end
        if tail_b > (total_frames // 2) and tail_b > head_a and (head_a + tail_b + 44 > total_frames) and tail_b <= total_frames:
            converted_tail = max(0, total_frames - tail_b)
            print(f"[LooperMinimaxH3] Interpreting tail_b={tail_b} as absolute end frame -> cut from end: {converted_tail}f")
            tail_b = converted_tail

        # Enforce minimum 44 frames active window (22 frames for Chunk A + 22 frames for Chunk B)
        max_possible_tail = max(0, total_frames - head_a - 44)
        if tail_b > max_possible_tail and total_frames >= 44:
            print(f"[LooperMinimaxH3] Clamping tail_b from {tail_b}f to {max_possible_tail}f to preserve 44f (14t) MiniMax context.")
            tail_b = max_possible_tail

        # Convert cuts to latent token indices snapped to causal 3D block boundaries (multiples of 5 tokens: 0, 5, 10...)
        # Guarantees that Chunk A and Chunk B always start at temporal convolution phase t % 5 == 0, eliminating
        # the t=0 causal activation surge (+43% brightness flare / blown out look) when decoded with VAEDecode.
        head_tokens = snap_to_minimax_token_boundary(minimax_frames_to_temporal(head_a)) if head_a > 0 else 0
        tail_tokens = snap_to_minimax_token_boundary(minimax_frames_to_temporal(tail_b)) if tail_b > 0 else 0

        min_required_tokens = head_tokens + tail_tokens + 14  # 7 tokens for A (22f) + 7 tokens for B (22f) minimum
        if total_tokens < min_required_tokens:
            raise ValueError(
                f"[LooperMinimaxH3] Latent has {total_tokens} temporal tokens (~{total_frames} frames), but at least "
                f"{min_required_tokens} tokens (~{minimax_temporal_to_frames(min_required_tokens)} frames) are required "
                f"for head_a ({head_tokens}t), tail_b ({tail_tokens}t) and 14 tokens (44f) minimum context. "
                f"Adjust head_a or tail_b."
            )

        end_token = total_tokens - tail_tokens
        active_tokens = end_token - head_tokens
        min_seam_token = head_tokens + 5
        max_seam_token = end_token - 5

        if seam_frame <= 0:
            # Auto split: align seam to causal 3D block boundary (multiple of 5 tokens: 0, 5, 10, 15, 20...)
            # Guarantees that Chunk B begins at causal phase t % 5 == 0, eliminating the t=0 causal activation surge
            # (+43% brightness flare / blown out look) when decoded with VAEDecode.
            ideal_seam = head_tokens + (active_tokens // 2)
            snapped = snap_to_minimax_token_boundary(ideal_seam)
            if snapped < min_seam_token:
                snapped_up = ((min_seam_token + 4) // 5) * 5
                seam_token = snapped_up if snapped_up <= max_seam_token else min_seam_token
            elif snapped > max_seam_token:
                snapped_down = (max_seam_token // 5) * 5
                seam_token = snapped_down if snapped_down >= min_seam_token else max_seam_token
            else:
                seam_token = snapped
        else:
            raw_seam = minimax_frames_to_temporal(seam_frame)
            snapped = snap_to_minimax_token_boundary(raw_seam)
            seam_token = max(min_seam_token, min(snapped, max_seam_token))

        actual_seam_frame = minimax_temporal_to_frames(seam_token)

        if pbar is not None:
            pbar.update_absolute(40, 100)

        # 3. Pure Homologous Latent Splitting (Identical to Video PreLooper: active[:split] and active[split:])
        # Chunk A is sliced with +2 tokens (length congruent to 5k + 2) so that when decoded individually by VAEDecode,
        # pad_tokens == 0, eliminating VAE temporal padding anomaly at the tail of Chunk A.
        # PostLooperLatent automatically deduplicates these 2 overlapping tokens upon assembly.
        chunk_a_end = min(total_tokens, seam_token + 2)
        chunk_latent_a = slice_minimax_latent(latent, head_tokens, chunk_a_end)
        chunk_latent_b = slice_minimax_latent(latent, seam_token, end_token)
        chunk_a_tokens = chunk_a_end - head_tokens
        chunk_b_tokens = end_token - seam_token

        # 4. Extract Guide Context Frames for MiniMax H3 Reference & Conditioning
        eff_guide_tokens = max(2, min(int(guide_tokens), chunk_a_tokens, chunk_b_tokens))
        eff_guide_frames = minimax_temporal_to_frames(eff_guide_tokens)

        t_dec = time.perf_counter()
        preview_frames = kwargs.get("preview_frames")
        decoded_full = None
        if preview_frames is not None and isinstance(preview_frames, torch.Tensor) and preview_frames.ndim >= 4:
            candidate = preview_frames[0] if preview_frames.ndim == 5 else preview_frames
            if candidate.shape[1] == target_h and candidate.shape[2] == target_w:
                decoded_full = candidate

        chunk_a_start_frame = minimax_temporal_to_frames(head_tokens)
        chunk_b_end_frame = minimax_temporal_to_frames(end_token)
        head_frame = chunk_a_start_frame
        end_frame = chunk_b_end_frame

        if decoded_full is not None:
            # Extract context guides directly from available full video frames (zero extra VAE decode cost)
            decoded_guide_a = decoded_full[head_frame : min(decoded_full.shape[0], head_frame + eff_guide_frames)].detach().clamp(0.0, 1.0)
            if decoded_guide_a.shape[0] < eff_guide_frames:
                decoded_guide_a = decoded_full[:min(eff_guide_frames, decoded_full.shape[0])].detach().clamp(0.0, 1.0)

            decoded_guide_b = decoded_full[max(0, end_frame - eff_guide_frames) : min(decoded_full.shape[0], end_frame)].detach().clamp(0.0, 1.0)
            if decoded_guide_b.shape[0] < eff_guide_frames:
                decoded_guide_b = decoded_full[-min(eff_guide_frames, decoded_full.shape[0]):].detach().clamp(0.0, 1.0)
        else:
            # Fast Selective Causal-Aligned Decode:
            # When total_tokens > 17, decode only the causal-aligned boundary clips (starting at t % 5 == 0)
            # instead of the entire video tensor, saving up to 75% VAE decode time and VRAM while
            # preserving 0.0000 causal activation flash.
            try:
                if total_tokens > 17:
                    a_slice_end = min(total_tokens, head_tokens + max(7, eff_guide_tokens))
                    b_causal_start = max(0, ((end_token - max(7, eff_guide_tokens)) // 5) * 5)
                    with torch.no_grad():
                        dec_a = vae.decode(video_tensor[:, :, head_tokens:a_slice_end].contiguous())
                        dec_b = vae.decode(video_tensor[:, :, b_causal_start:end_token].contiguous())
                    if dec_a.ndim == 5:
                        dec_a = dec_a[0]
                    if dec_b.ndim == 5:
                        dec_b = dec_b[0]
                    decoded_guide_a = dec_a[:min(eff_guide_frames, dec_a.shape[0])].detach().clamp(0.0, 1.0)
                    decoded_guide_b = dec_b[-min(eff_guide_frames, dec_b.shape[0]):].detach().clamp(0.0, 1.0)
                else:
                    with torch.no_grad():
                        decoded_video = vae.decode(video_tensor)
                    if decoded_video.ndim == 5:
                        decoded_video = decoded_video[0]
                    decoded_guide_a = decoded_video[head_frame : min(decoded_video.shape[0], head_frame + eff_guide_frames)].detach().clamp(0.0, 1.0)
                    if decoded_guide_a.shape[0] < eff_guide_frames:
                        decoded_guide_a = decoded_video[:min(eff_guide_frames, decoded_video.shape[0])].detach().clamp(0.0, 1.0)
                    decoded_guide_b = decoded_video[max(0, end_frame - eff_guide_frames) : min(decoded_video.shape[0], end_frame)].detach().clamp(0.0, 1.0)
                    if decoded_guide_b.shape[0] < eff_guide_frames:
                        decoded_guide_b = decoded_video[-min(eff_guide_frames, decoded_video.shape[0]):].detach().clamp(0.0, 1.0)
            except Exception as e:
                # Direct guide slice decode fallback
                guide_latent_a = slice_minimax_latent(chunk_latent_a, 0, eff_guide_tokens)
                guide_latent_b = slice_minimax_latent(chunk_latent_b, chunk_b_tokens - eff_guide_tokens, chunk_b_tokens)
                g_video_a, _ = extract_video_audio_latents(guide_latent_a)
                with torch.no_grad():
                    decoded_guide_a = vae.decode(g_video_a)
                if decoded_guide_a.ndim == 5:
                    decoded_guide_a = decoded_guide_a[0]
                decoded_guide_a = decoded_guide_a.detach().clamp(0.0, 1.0)[:eff_guide_frames]
                g_video_b, _ = extract_video_audio_latents(guide_latent_b)
                with torch.no_grad():
                    decoded_guide_b = vae.decode(g_video_b)
                if decoded_guide_b.ndim == 5:
                    decoded_guide_b = decoded_guide_b[0]
                decoded_guide_b = decoded_guide_b.detach().clamp(0.0, 1.0)[-eff_guide_frames:]

        elapsed_dec = time.perf_counter() - t_dec

        # Reference images for Qwen vision tokenizer (extracted from clean continuous video):
        # image_0 (<Picture 1>): last frame of Chunk B context (origin of transition, anchors frame_idx = 0)
        last_b_idx = max(0, decoded_guide_b.shape[0] - 1)
        image_0 = decoded_guide_b[last_b_idx : last_b_idx + 1]

        # image_1 (<Picture 2>): first frame of Chunk A context (destination of transition)
        image_1 = decoded_guide_a[0:1]

        # 5. Audio handling and slicing (matching cuts 1:1)
        video_audio = kwargs.get("load_audio")
        fps = 24.0
        chunk_audio_a = None
        chunk_audio_b = None
        guide_a_audio = None
        guide_b_audio = None

        if video_audio is not None and "waveform" in video_audio and video_audio["waveform"].shape[-1] > 0:
            sr = video_audio["sample_rate"]
            waveform = video_audio["waveform"]
            total_samples = waveform.shape[-1]

            start_samp = max(0, int(round(chunk_a_start_frame * sr / fps)))
            seam_samp = min(total_samples, int(round(actual_seam_frame * sr / fps)))
            end_samp = min(total_samples, int(round(chunk_b_end_frame * sr / fps)))

            chunk_audio_a = {
                "waveform": waveform[..., start_samp:seam_samp],
                "sample_rate": sr
            }
            chunk_audio_b = {
                "waveform": waveform[..., seam_samp:end_samp],
                "sample_rate": sr
            }

            guide_a_end_samp = min(total_samples, int(round((chunk_a_start_frame + eff_guide_frames) * sr / fps)))
            guide_b_start_samp = max(0, int(round((chunk_b_end_frame - eff_guide_frames) * sr / fps)))

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

        if pbar is not None:
            pbar.update_absolute(80, 100)

        # 7. Total generation length snapped to native MiniMax H3 17k + 5 boundary
        duration_length = snap_to_minimax_generation_length(
            trans_duration + 2 * eff_guide_frames,
            min_frames=2 * eff_guide_frames + 5
        )

        # 8. Directly execute MiniMaxH3ReferenceToVideo to generate base conditioning & latent
        ref_audios = None
        if audio_vae is not None and guide_b_audio is not None and guide_b_audio["waveform"].shape[-1] > 0:
            ref_audios = {"ref_audio_0": guide_b_audio, "ref_audio_1": guide_a_audio}

        t_ref = time.perf_counter()
        positive, trans_empty_latent = call_minimax_h3_reference(
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

        if positive is None or trans_empty_latent is None:
            raise RuntimeError(
                "[LooperMinimaxH3] Failed to generate conditioning and latent using MiniMaxH3ReferenceToVideo. "
                "Ensure MiniMax H3 models and comfy_extras are properly installed."
            )

        if isinstance(trans_empty_latent, dict):
            trans_empty_latent["guide_overlap_tokens"] = eff_guide_tokens

        if pbar is not None:
            pbar.update_absolute(90, 100)

        # 9. Inject MiniMax H3 Continuation Guides into positive conditioning
        # Uses native latent keyframes (ComfyUI PackedLayout) from chunk_b tail and chunk_a head
        # with zero VAE re-encode loss and full unhindered ODE flow velocity.
        positive = apply_minimax_h3_continuation_guides(
            positive=positive,
            trans_empty_latent=trans_empty_latent,
            chunk_latent_b=chunk_latent_b,
            chunk_latent_a=chunk_latent_a,
            vae=vae,
            audio_vae=audio_vae,
            guide_b_audio=guide_b_audio,
            guide_a_audio=guide_a_audio,
            decoded_guide_b=decoded_guide_b,
            decoded_guide_a=decoded_guide_a,
            k_tokens=eff_guide_tokens
        )

        # 10. Integrated Continuity Patcher on optional `model` input
        patched_model = None
        if model is not None:
            patched_model = patch_minimax_h3_continuity(
                model=model,
                chunk_latent_b=chunk_latent_b,
                chunk_latent_a=chunk_latent_a,
                k_tokens=eff_guide_tokens,
                blend_tokens=blend_tokens,
                strength=continuity_strength,
                pad_clips=pad_clips,
                sigma_schedule=sigma_schedule
            )

        if pbar is not None:
            pbar.update_absolute(100, 100)

        t_total = time.perf_counter() - t_start

        # Detailed Metrics Summary
        print("-" * 65)
        print(f"[LooperMinimaxH3] >>> PREPROCESS LATENT METRICS SUMMARY:")
        print(f"[LooperMinimaxH3]  - Latent Dimensions: {total_tokens}t (~{total_frames}f) @ {target_w}x{target_h}")
        print(f"[LooperMinimaxH3]  - Guides Decoded ({2 * eff_guide_frames}f): {elapsed_dec:.2f}s")
        print(f"[LooperMinimaxH3]  - Chunk Latent A: {chunk_a_tokens}t (~{minimax_temporal_to_frames(chunk_a_tokens)}f, tokens {head_tokens}..{seam_token - 1})")
        print(f"[LooperMinimaxH3]  - Seam Split Token: Token {seam_token} (~Frame {actual_seam_frame})")
        print(f"[LooperMinimaxH3]  - Chunk Latent B: {chunk_b_tokens}t (~{minimax_temporal_to_frames(chunk_b_tokens)}f, tokens {seam_token}..{end_token - 1})")
        print(f"[LooperMinimaxH3]  - Reference & Conditioning: {duration_length}f in {elapsed_ref:.2f}s")
        if model is not None:
            print(f"[LooperMinimaxH3]  - Integrated Model Patch: strength={continuity_strength:.2f}, pad_clips={pad_clips}, sigma_schedule={sigma_schedule}")
        print(f"[LooperMinimaxH3]  - TOTAL PREPROCESS TIME: {t_total:.2f}s")
        print("=" * 65 + "\n")

        return (
            chunk_latent_a,
            chunk_latent_b,
            positive,
            trans_empty_latent,
            out_audio_a,
            out_audio_b,
            patched_model
        )


class PreLooperLatentPreviewMinimaxNode:
    """
    Pre Looper Latent Preview Node for MiniMax H3 (PreLooperLatentPreviewMinimax).
    Positioned prior to PreLooperLatentMinimax in pure latent workflows:
    - Receives incoming latent tensor directly from base generator
    - Decodes a fast, lightweight miniature preview using the provided VAE
    - Provides interactive timeline panel (scrubber, head cut, tail cut, seam adjustment, circular loop playback)
    - Passes through latent intact (no re-encoding / no generation drift)
    - Outputs head_a, tail_b, seam_frame and preview_frames for downstream PreLooperLatentMinimax
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Incoming video latent to preview and set cut points"}),
                "vae": ("VAE", {"tooltip": "VAE to decode miniature preview frames"}),
                "head_a": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Frames to cut from the start of Chunk A"}),
                "tail_b": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Frames to cut from the end of Chunk B"}),
                "seam_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "Frame where new video begins (cut between Chunk A and B). 0 = auto middle split"}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0, "tooltip": "Framerate for the preview playback"}),
                "preview_max_dim": ("INT", {"default": 384, "min": 128, "max": 768, "step": 32, "tooltip": "Maximum spatial dimension for lightweight miniature preview"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("latent", "head_a", "tail_b", "seam_frame", "preview_frames")
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "LooperMinimaxH3"

    def process(self, latent: dict, vae, head_a: int = 0, tail_b: int = 0, seam_frame: int = 0,
                fps: float = 24.0, preview_max_dim: int = 384, **kwargs):
        t_start = time.perf_counter()
        pbar = ProgressBar(100) if ProgressBar is not None else None
        if pbar is not None:
            pbar.update_absolute(10, 100)

        print("\n" + "=" * 65)
        print(f"[LooperMinimaxH3] >>> PRE-LOOPER LATENT PREVIEW START")
        print("=" * 65)

        video_tensor, _ = extract_video_audio_latents(latent)
        if video_tensor is None or video_tensor.ndim != 5:
            raise ValueError(f"[LooperMinimaxH3] Expected a 5D video latent tensor [B, C, T, H, W], got {video_tensor.shape if video_tensor is not None else None}")

        total_tokens = video_tensor.shape[2]
        total_frames = minimax_temporal_to_frames(total_tokens)

        # Smart disambiguation: detect if tail_b was passed as absolute frame index
        head_a = max(0, head_a)
        tail_b = max(0, tail_b)
        if tail_b > (total_frames // 2) and tail_b > head_a and (head_a + tail_b + 44 > total_frames) and tail_b <= total_frames:
            converted_tail = max(0, total_frames - tail_b)
            print(f"[LooperMinimaxH3] Interpreting tail_b={tail_b} as absolute end frame -> cut from end: {converted_tail}f")
            tail_b = converted_tail

        max_possible_tail = max(0, total_frames - head_a - 44)
        if tail_b > max_possible_tail and total_frames >= 44:
            print(f"[LooperMinimaxH3] Clamping tail_b from {tail_b}f to {max_possible_tail}f to preserve 44f (14t) MiniMax context.")
            tail_b = max_possible_tail

        if pbar is not None:
            pbar.update_absolute(30, 100)

        # 1. Decode latent via VAE
        try:
            with torch.no_grad():
                decoded = vae.decode(video_tensor)
            if decoded.ndim == 5:
                decoded = decoded[0]  # Shape: [T, H, W, C]
        except Exception as e:
            print(f"[LooperMinimaxH3] VAE decode failed during preview: {e}")
            raise RuntimeError(f"Error decoding latent with VAE for miniature preview: {e}")

        if pbar is not None:
            pbar.update_absolute(70, 100)

        # 2. Downscale decoded frames if exceeding preview_max_dim to preserve memory
        num_frames, h, w, c = decoded.shape
        if h > preview_max_dim or w > preview_max_dim:
            scale = min(1.0, preview_max_dim / max(h, w))
            new_h = max(64, (int(round(h * scale)) // 2) * 2)
            new_w = max(64, (int(round(w * scale)) // 2) * 2)
            t_perm = decoded.permute(0, 3, 1, 2)
            t_down = torch.nn.functional.interpolate(t_perm, size=(new_h, new_w), mode="bilinear", align_corners=False)
            preview_frames = t_down.permute(0, 2, 3, 1).clamp(0.0, 1.0).cpu()
        else:
            preview_frames = decoded.clamp(0.0, 1.0).cpu()

        # 3. Encode lightweight miniature MP4 preview video for ComfyUI web UI
        preview_meta = save_preview_video(preview_frames, fps=fps, max_dim=preview_max_dim)

        if pbar is not None:
            pbar.update_absolute(100, 100)

        t_elapsed = time.perf_counter() - t_start
        print(f"[LooperMinimaxH3] Preview generated: {num_frames} frames ({preview_frames.shape[2]}x{preview_frames.shape[1]}) in {t_elapsed:.2f}s")
        print(f"[LooperMinimaxH3] Cuts configured -> Head: {head_a}f, Tail: {tail_b}f, Seam: {seam_frame}f")
        print("=" * 65 + "\n")

        ui_output = {"videos": [preview_meta]} if preview_meta else {}
        return {
            "ui": ui_output,
            "result": (latent, int(head_a), int(tail_b), int(seam_frame), preview_frames)
        }

