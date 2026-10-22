import math
import torch
from .looper_utils import resize_image_tensor, slice_audio, concat_audio, interpolate_circular_loop

class PostLooperVideoMinimaxNode:
    """
    Post Looper Video node for MiniMax H3 (postlooper_video).
    Trims the guide overlap (22 frames at start and 22 frames at end) of the generated transition video
    and joins it in seamless circular order with chunk_b and chunk_a using PyTorch tensors.
    Also handles audio stitching circularly with smooth crossfades if audio inputs are connected,
    and supports optional circular loop frame interpolation (including last-to-first frame).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trans_image": ("IMAGE", {"tooltip": "Generated transition video from MiniMax H3 (output of VAEDecode)"}),
                "chunk_b": ("IMAGE", {"tooltip": "Second half of original video (output 'chunk_b' from Pre Looper)"}),
                "chunk_a": ("IMAGE", {"tooltip": "First half of original video (output 'chunk_a' from Pre Looper)"}),
            },
            "optional": {
                "trans_audio": ("AUDIO", {"tooltip": "Optional audio of transition video (output of VAEDecodeAudio)"}),
                "chunk_audio_b": ("AUDIO", {"tooltip": "Audio of Chunk B (output 'chunk_audio_b' from Pre Looper)"}),
                "chunk_audio_a": ("AUDIO", {"tooltip": "Audio of Chunk A (output 'chunk_audio_a' from Pre Looper)"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1, "tooltip": "Framerate for accurate audio trimming (default 24.0)"}),
                "smooth_boundaries": ("BOOLEAN", {"default": True, "tooltip": "Harmonize boundary transition frames at Seam 1 and Seam 2 with zero-jerk velocity continuity"}),
                "interpolate_multiplier": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "Circular frame interpolation multiplier (1 = disabled/original FPS, 2 = 2x FPS including last-to-first loop closure)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("image", "audio", "length")
    FUNCTION = "process"
    CATEGORY = "LooperMinimaxH3"

    def process(self, trans_image: torch.Tensor, chunk_b: torch.Tensor, chunk_a: torch.Tensor,
                trans_audio=None, chunk_audio_b=None, chunk_audio_a=None,
                fps: float = 24.0, smooth_boundaries: bool = True,
                interpolate_multiplier: int = 1, **kwargs):
        
        # Support backwards-compatible input slot names
        if chunk_audio_b is None:
            chunk_audio_b = kwargs.get("chunk_b_audio")
        if chunk_audio_a is None:
            chunk_audio_a = kwargs.get("chunk_a_audio")

        # In MiniMax H3 (matching the reference workflow), loop order is circular B -> Trans -> A:
        # [chunk_b] -> [trimmed_trans] -> [chunk_a].
        guide_overlap = kwargs.get("guide_overlap", 22)
        loop_order = kwargs.get("loop_order", "B -> Trans -> A (Recommended)")

        trans_total_frames = trans_image.shape[0]

        # 1. Trim guide overlap frames from transition video
        if guide_overlap > 0:
            min_required = guide_overlap * 2
            if trans_total_frames <= min_required:
                raise ValueError(
                    f"[LooperMinimaxH3] trans_image has only {trans_total_frames} frames. "
                    f"It must have more than 2 * guide_overlap ({min_required} frames). "
                    f"Ensure trans_duration in Pre Looper is set appropriately."
                )
            trans_guide_b_last = trans_image[guide_overlap - 1:guide_overlap].clone()
            trans_guide_a_first = trans_image[-guide_overlap:trans_total_frames - guide_overlap + 1].clone()
            trans_trimmed = trans_image[guide_overlap:-guide_overlap].clone()
        else:
            trans_guide_b_last = None
            trans_guide_a_first = None
            trans_trimmed = trans_image

        trimmed_count = trans_trimmed.shape[0]
        target_h = chunk_a.shape[1]
        target_w = chunk_a.shape[2]

        # 2. Ensure spatial resolutions match exactly
        if trans_trimmed.shape[1] != target_h or trans_trimmed.shape[2] != target_w:
            print(
                f"[LooperMinimaxH3] Resizing trans_trimmed ({trans_trimmed.shape[2]}x{trans_trimmed.shape[1]}) "
                f"to match chunk_a ({target_w}x{target_h})"
            )
            trans_trimmed = resize_image_tensor(trans_trimmed, target_w, target_h)

        if trans_guide_b_last is not None and (trans_guide_b_last.shape[1] != target_h or trans_guide_b_last.shape[2] != target_w):
            trans_guide_b_last = resize_image_tensor(trans_guide_b_last, target_w, target_h)
        if trans_guide_a_first is not None and (trans_guide_a_first.shape[1] != target_h or trans_guide_a_first.shape[2] != target_w):
            trans_guide_a_first = resize_image_tensor(trans_guide_a_first, target_w, target_h)

        if chunk_b.shape[1] != target_h or chunk_b.shape[2] != target_w:
            chunk_b = resize_image_tensor(chunk_b, target_w, target_h)

        # Auto-detect if video inputs were plugged inverted (chunk_a into chunk_b slot)
        try:
            if chunk_b.shape[0] != chunk_a.shape[0] or not torch.allclose(chunk_b[0], chunk_a[0]):
                diff_normal = torch.mean(torch.abs(chunk_b[-1] - trans_trimmed[0])) + torch.mean(torch.abs(trans_trimmed[-1] - chunk_a[0]))
                diff_swapped = torch.mean(torch.abs(chunk_a[-1] - trans_trimmed[0])) + torch.mean(torch.abs(trans_trimmed[-1] - chunk_b[0]))
                if diff_swapped + 0.08 < diff_normal:
                    print("[LooperMinimaxH3] Notice: Auto-correcting inverted chunk inputs (swapped chunk_a <-> chunk_b)")
                    chunk_b, chunk_a = chunk_a, chunk_b
        except Exception:
            pass

        # 2b. Boundary velocity offset cancellation at Seam 1 (Chunk B -> Transition) and Seam 2 (Transition -> Chunk A):
        # Eliminates the boundary reconstruction gap between original camera frames and the diffusion guide frames,
        # providing 100% natural, continuous forward velocity into and out of the transition.
        if smooth_boundaries and trans_guide_b_last is not None and chunk_b.shape[0] > 0 and trans_trimmed.shape[0] > 0:
            if trans_image.std() > 1e-3 and chunk_b.std() > 1e-3:
                E_seam1 = trans_guide_b_last - chunk_b[-1:]
                K1 = min(5, max(1, trans_trimmed.shape[0] // 2))
                for k in range(K1):
                    w = 0.5 * (1.0 + math.cos(math.pi * k / K1))
                    trans_trimmed[k:k+1] = (trans_trimmed[k:k+1] - E_seam1 * w).clamp(0.0, 1.0)
                if trans_guide_a_first is not None and chunk_a.shape[0] > 0 and chunk_a.std() > 1e-3:
                    E_seam2 = trans_guide_a_first - chunk_a[0:1]
                    for k in range(K1):
                        w = 0.5 * (1.0 + math.cos(math.pi * k / K1))
                        idx = trans_trimmed.shape[0] - 1 - k
                        trans_trimmed[idx:idx+1] = (trans_trimmed[idx:idx+1] - E_seam2 * w).clamp(0.0, 1.0)

        # Auto-detect if audio inputs were plugged inverted (A into B slot)
        if chunk_audio_b is not None and chunk_audio_a is not None:
            wb = chunk_audio_b.get("waveform", torch.empty(0))
            wa = chunk_audio_a.get("waveform", torch.empty(0))
            if wb.shape[-1] > 0 and wa.shape[-1] > 0:
                sr = chunk_audio_b.get("sample_rate", 44100)
                dur_b_sec = wb.shape[-1] / sr
                dur_a_sec = wa.shape[-1] / sr
                expected_b_sec = chunk_b.shape[0] / max(1.0, fps)
                expected_a_sec = chunk_a.shape[0] / max(1.0, fps)
                err_normal = abs(dur_b_sec - expected_b_sec) + abs(dur_a_sec - expected_a_sec)
                err_swapped = abs(dur_b_sec - expected_a_sec) + abs(dur_a_sec - expected_b_sec)
                if err_swapped + 0.15 < err_normal:
                    print("[LooperMinimaxH3] Auto-correcting swapped audio connections (chunk_audio_a <-> chunk_audio_b)")
                    chunk_audio_b, chunk_audio_a = chunk_audio_a, chunk_audio_b

        # 3. Concatenate video in circular seamless loop order:
        if "A -> B -> Trans" in loop_order:
            final_loop_video = torch.cat([chunk_a, chunk_b, trans_trimmed], dim=0)
        else:
            final_loop_video = torch.cat([chunk_b, trans_trimmed, chunk_a], dim=0)

        # 3b. Optional Circular Frame Interpolation across the whole loop (including last-to-first frame)
        if interpolate_multiplier > 1:
            interp_method = kwargs.get("interpolation_method", "FILM")
            final_loop_video, _ = interpolate_circular_loop(
                final_loop_video,
                multiplier=int(interpolate_multiplier),
                fps=fps,
                method=interp_method
            )

        total_loop_frames = final_loop_video.shape[0]

        # 4. Concatenate audio in matching circular order:
        final_loop_audio = None
        empty_audio = {"waveform": torch.zeros((1, 2, 0), dtype=torch.float32), "sample_rate": 44100}

        has_b_audio = (chunk_audio_b is not None and chunk_audio_b.get("waveform", torch.empty(0)).shape[-1] > 0)
        has_a_audio = (chunk_audio_a is not None and chunk_audio_a.get("waveform", torch.empty(0)).shape[-1] > 0)
        has_trans_audio = (trans_audio is not None and trans_audio.get("waveform", torch.empty(0)).shape[-1] > 0)

        if has_b_audio or has_a_audio or has_trans_audio:
            trans_audio_trimmed = None
            if has_trans_audio and guide_overlap > 0:
                overlap_sec = guide_overlap / max(1.0, fps)
                sr = trans_audio["sample_rate"]
                w = trans_audio["waveform"]
                total_samples = w.shape[-1]
                trim_samp = int(round(overlap_sec * sr))
                if total_samples > 2 * trim_samp:
                    trans_audio_trimmed = {
                        "waveform": w[..., trim_samp : total_samples - trim_samp],
                        "sample_rate": sr
                    }
                else:
                    trans_audio_trimmed = trans_audio
            elif has_trans_audio:
                trans_audio_trimmed = trans_audio

            audio_parts = []
            if "A -> B -> Trans" in loop_order:
                if has_a_audio:
                    audio_parts.append(chunk_audio_a)
                if has_b_audio:
                    audio_parts.append(chunk_audio_b)
                if trans_audio_trimmed is not None:
                    audio_parts.append(trans_audio_trimmed)
            else:
                if has_b_audio:
                    audio_parts.append(chunk_audio_b)
                if trans_audio_trimmed is not None:
                    audio_parts.append(trans_audio_trimmed)
                if has_a_audio:
                    audio_parts.append(chunk_audio_a)

            if audio_parts:
                final_loop_audio = concat_audio(audio_parts, crossfade_sec=0.02)

        if final_loop_audio is None:
            final_loop_audio = empty_audio

        print(
            f"[LooperMinimaxH3] Post Looper Video joined: "
            f"Chunk B ({chunk_b.shape[0]}f) + Transition ({trimmed_count}f) + Chunk A ({chunk_a.shape[0]}f) "
            f"= Total Loop: {total_loop_frames} frames ({target_w}x{target_h})"
        )

        return (final_loop_video, final_loop_audio, total_loop_frames)

# Alias for backwards compatibility with previous versions
PostLooperMinimaxNode = PostLooperVideoMinimaxNode
