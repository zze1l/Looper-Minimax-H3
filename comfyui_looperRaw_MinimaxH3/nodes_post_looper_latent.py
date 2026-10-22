import math
import time
import torch

from .looper_utils import (
    minimax_temporal_to_frames,
    minimax_frames_to_temporal,
    extract_video_audio_latents,
    create_latent_dict,
    harmonize_latent_manifold,
    smooth_latent_boundaries,
    wrap_pad_periodic,
    MINIMAX_CLIP_LATENTS,
    MINIMAX_VIDEO_TAIL,
    slice_audio,
    concat_audio,
)

class PostLooperLatentMinimaxNode:
    """
    Post Looper Latent node for MiniMax H3 (PostLooperLatentMinimax / postlooper_latent).
    Homologous to PostLooperVideoMinimaxNode, operating natively in latent space:
    - Trims the 7-token guide overlap (22 frames at start and 22 frames at end) of trans_latent
    - Supports both frame count (e.g. 22) and token count (e.g. 7) with automatic disambiguation
    - Joins in seamless circular loop order in VRAM: [chunk_latent_a] + [chunk_latent_b] + [trans_latent]
    - Assembles audio circularly with smooth crossfades if audio inputs are connected
    - Clean contract identical to PostLooperVideoMinimaxNode (LATENT, AUDIO, INT)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trans_latent": ("LATENT", {"tooltip": "Generated transition latent from MiniMax H3 (output of SamplerCustomAdvanced)"}),
                "chunk_latent_b": ("LATENT", {"tooltip": "Second half of original video in latent space (output 'chunk_latent_b' from Pre Looper Latent)"}),
                "chunk_latent_a": ("LATENT", {"tooltip": "First half of original video in latent space (output 'chunk_latent_a' from Pre Looper Latent)"}),
            },
            "optional": {
                "trans_audio": ("AUDIO", {"tooltip": "Optional audio of transition video"}),
                "chunk_audio_b": ("AUDIO", {"tooltip": "Audio of Chunk B (output 'chunk_audio_b' from Pre Looper Latent)"}),
                "chunk_audio_a": ("AUDIO", {"tooltip": "Audio of Chunk A (output 'chunk_audio_a' from Pre Looper Latent)"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1, "tooltip": "Framerate for audio trimming (default 24.0)"}),
                "smooth_boundaries": ("BOOLEAN", {"default": True, "tooltip": "Harmonize latent boundary tokens at Seam 1 (Chunk B -> Trans) with zero-jerk velocity continuity"}),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO", "INT")
    RETURN_NAMES = ("latent", "audio", "length")
    FUNCTION = "process"
    CATEGORY = "LooperMinimaxH3"

    def process(self, trans_latent: dict, chunk_latent_b: dict, chunk_latent_a: dict,
                trans_audio=None, chunk_audio_b=None, chunk_audio_a=None,
                fps: float = 24.0, smooth_boundaries: bool = True, **kwargs):
        
        t_start = time.perf_counter()

        # Support backwards-compatible input slot names
        if chunk_audio_b is None:
            chunk_audio_b = kwargs.get("chunk_b_audio")
        if chunk_audio_a is None:
            chunk_audio_a = kwargs.get("chunk_a_audio")

        # In MiniMax H3 (matching the reference workflow), loop order is circular B -> Trans -> A:
        # [chunk_b] -> [trimmed_trans] -> [chunk_a].
        # When Chunk A ends, it loops back to Chunk B, which is the 100% continuous original video seam.
        default_guide_tokens = trans_latent.get("guide_overlap_tokens", 7) if isinstance(trans_latent, dict) else 7
        guide_overlap_tokens = kwargs.get("guide_overlap_tokens", kwargs.get("guide_overlap", default_guide_tokens))
        loop_order = kwargs.get("loop_order", "B -> Trans -> A (Recommended)")
        circular_pad_clips = int(kwargs.get("circular_pad_clips", 0))

        # 1. Extract latents
        trans_video, trans_audio_lat = extract_video_audio_latents(trans_latent)
        chunk_b_video, chunk_b_audio_lat = extract_video_audio_latents(chunk_latent_b)
        chunk_a_video, chunk_a_audio_lat = extract_video_audio_latents(chunk_latent_a)

        trans_tokens = trans_video.shape[2]

        # 2. Smart Disambiguation for Guide Overlap:
        # In MiniMax H3, guides are 22 frames = 7 latent tokens.
        # If the parameter is passed >= 14 (e.g. 22, as in standard video postlooper),
        # interpret it as frames and convert to temporal tokens via minimax_frames_to_temporal().
        # If passed < 14 (e.g. 7 tokens), treat it directly as tokens.
        if guide_overlap_tokens >= 14:
            actual_guide_tokens = minimax_frames_to_temporal(guide_overlap_tokens)
            overlap_frames = guide_overlap_tokens
            print(f"[LooperMinimaxH3] Interpreting guide_overlap={guide_overlap_tokens}f as frames -> {actual_guide_tokens} latent tokens.")
        else:
            actual_guide_tokens = max(0, guide_overlap_tokens)
            overlap_frames = 22 if actual_guide_tokens == 7 else minimax_temporal_to_frames(actual_guide_tokens)

        # 3. Trim guide overlap tokens from transition latent
        # Exactly matching the reference workflow, trimming actual_guide_tokens (7 tokens / 22 frames)
        # on both ends eliminates any duplicate frames while preserving exact temporal continuity.
        start_trim = actual_guide_tokens
        end_trim = actual_guide_tokens

        if actual_guide_tokens > 0:
            min_required = start_trim + end_trim
            if trans_tokens <= min_required:
                req_frames = minimax_temporal_to_frames(min_required)
                trans_frames = minimax_temporal_to_frames(trans_tokens)
                raise ValueError(
                    f"[LooperMinimaxH3] trans_latent has only {trans_tokens} temporal tokens (~{trans_frames} frames). "
                    f"It must have more than {min_required} guide tokens (~{req_frames} frames). "
                    f"Ensure trans_duration in Pre Looper Latent is set appropriately."
                )
            trimmed_trans_video = trans_video[:, :, start_trim:trans_tokens - end_trim, :, :].clone()

            # Boundary velocity offset cancellation at Seam 1 (Chunk B -> Transition):
            # Eliminates the latent reconstruction error between the diffusion model's guide token
            # (trans_video[:, :, start_trim - 1]) and the original video latent (chunk_b_video[:, :, -1]),
            # providing 100% natural, continuous forward velocity into the transition without any sudden jump or stutter.
            if smooth_boundaries and chunk_b_video.shape[2] > 0 and trimmed_trans_video.shape[2] > 0:
                if trans_video.shape[3:] == chunk_b_video.shape[3:] and trans_video.std() > 1e-3 and chunk_b_video.std() > 1e-3:
                    E_seam = trans_video[:, :, start_trim - 1:start_trim, :, :] - chunk_b_video[:, :, -1:, :, :]
                    K = min(4, trimmed_trans_video.shape[2])
                    for k in range(K):
                        w = 0.5 * (1.0 + math.cos(math.pi * k / K))
                        trimmed_trans_video[:, :, k:k+1, :, :] -= E_seam * w

            if trans_audio_lat is not None:
                a_start = int(round(start_trim * (40/24)))
                a_end = int(round(end_trim * (40/24)))
                trimmed_trans_audio_lat = trans_audio_lat[..., a_start:trans_audio_lat.shape[-1] - a_end] if trans_audio_lat.shape[-1] > (a_start + a_end) else None
            else:
                trimmed_trans_audio_lat = None
        else:
            trimmed_trans_video = trans_video.clone()
            trimmed_trans_audio_lat = trans_audio_lat

        trimmed_tokens = trimmed_trans_video.shape[2]
        b_tokens = chunk_b_video.shape[2]
        a_tokens = chunk_a_video.shape[2]

        # 4. Deduplicate 2-token VAE extension from Chunk A if present:
        # PreLooperLatent extends Chunk A by 2 tokens so its isolated length is congruent to (5k + 2),
        # ensuring pad_tokens == 0 when Chunk A is decoded individually by VAEDecode.
        # Upon loop assembly in PostLooperLatent, these 2 overlapping tokens are deduplicated
        # to ensure seamless 1:1 bit-exact reconstruction of the continuous original video.
        if a_tokens >= 4 and b_tokens >= 4:
            if torch.allclose(chunk_a_video[:, :, -2:], chunk_b_video[:, :, :2], atol=1e-3):
                print(f"[LooperMinimaxH3] Deduplicating 2-token VAE extension from Chunk A ({a_tokens}t -> {a_tokens - 2}t) for seamless seam assembly.")
                chunk_a_video = chunk_a_video[:, :, :-2]
                a_tokens = chunk_a_video.shape[2]
                if chunk_a_audio_lat is not None and chunk_a_audio_lat.shape[-1] > 2:
                    chunk_a_audio_lat = chunk_a_audio_lat[..., :-2]
            elif torch.allclose(chunk_b_video[:, :, -2:], chunk_a_video[:, :, :2], atol=1e-3):
                print(f"[LooperMinimaxH3] Detected inverted chunk connection. Swapping Chunk A and Chunk B and deduplicating 2 tokens...")
                chunk_a_video, chunk_b_video = chunk_b_video[:, :, :-2], chunk_a_video
                a_tokens = chunk_a_video.shape[2]
                b_tokens = chunk_b_video.shape[2]
                if chunk_b_audio_lat is not None:
                    chunk_a_audio_lat, chunk_b_audio_lat = chunk_b_audio_lat[..., :-2], chunk_a_audio_lat

        # 5. Harmonize transition latent manifold statistics if requested via kwargs
        if kwargs.get("harmonize_manifold", False):
            trimmed_trans_video = harmonize_latent_manifold(trimmed_trans_video, chunk_b_video)

        # 6. Concatenate video latents cleanly in circular loop order:
        # Default is "B -> Trans -> A (Recommended)":
        # [chunk_b] -> [trimmed_trans] -> [chunk_a]
        # Preserves 100% bit-exact original video continuity across the loop boundary (chunk_a -> chunk_b).
        if "A -> B -> Trans" in loop_order:
            joined_video = torch.cat([chunk_a_video, chunk_b_video, trimmed_trans_video], dim=2).clone().contiguous()
        else:
            joined_video = torch.cat([chunk_b_video, trimmed_trans_video, chunk_a_video], dim=2).clone().contiguous()

        # 7. Align total joined loop length to 5k + 2 tokens (zero VAE pad_tokens) if requested via kwargs:
        if kwargs.get("align_vae_tokens", False):
            rem = (joined_video.shape[2] - 2) % 5
            if rem != 0 and trimmed_trans_video.shape[2] > rem + 2:
                trimmed_trans_video = trimmed_trans_video[:, :, :-rem]
                trimmed_tokens = trimmed_trans_video.shape[2]
                if trimmed_trans_audio_lat is not None and trimmed_trans_audio_lat.shape[-1] > rem:
                    trimmed_trans_audio_lat = trimmed_trans_audio_lat[..., :-rem]
                if "A -> B -> Trans" in loop_order:
                    joined_video = torch.cat([chunk_a_video, chunk_b_video, trimmed_trans_video], dim=2).clone().contiguous()
                else:
                    joined_video = torch.cat([chunk_b_video, trimmed_trans_video, chunk_a_video], dim=2).clone().contiguous()
                print(f"[LooperMinimaxH3] Aligned final joined loop latent to 5k+2 tokens ({joined_video.shape[2]}t) for zero VAE pad_tokens.")

        # 7b. Optional Circular Latent Padding (inspired by MiniMaxH3LoopDecode):
        # Wraps `circular_pad_clips` (5 tokens = 17f per clip) on both sides so downstream VAEDecode
        # never starts or ends inside the active loop.
        if circular_pad_clips > 0 and joined_video.shape[2] >= MINIMAX_CLIP_LATENTS:
            pv = circular_pad_clips * MINIMAX_CLIP_LATENTS
            joined_video = wrap_pad_periodic(joined_video, pv, pv + MINIMAX_VIDEO_TAIL, dim=2).contiguous()

        total_tokens = joined_video.shape[2]
        total_length_frames = minimax_temporal_to_frames(total_tokens)

        # 6. Concatenate audio latents if present in NestedTensor
        joined_audio_lat = None
        if chunk_b_audio_lat is not None and chunk_a_audio_lat is not None:
            if "A -> B -> Trans" in loop_order:
                audios_to_join = [chunk_a_audio_lat, chunk_b_audio_lat]
                if trimmed_trans_audio_lat is not None:
                    audios_to_join.append(trimmed_trans_audio_lat)
            else:
                audios_to_join = [chunk_b_audio_lat]
                if trimmed_trans_audio_lat is not None:
                    audios_to_join.append(trimmed_trans_audio_lat)
                audios_to_join.append(chunk_a_audio_lat)
            try:
                joined_audio_lat = torch.cat(audios_to_join, dim=-1)
            except Exception as e:
                print(f"[LooperMinimaxH3] Note: audio latent concatenation skipped ({e})")

        out_latent = create_latent_dict(joined_video, joined_audio_lat)

        # 7. Process external Audio Waveforms with seamless crossfades
        out_audio = {"waveform": torch.zeros((1, 2, 0), dtype=torch.float32), "sample_rate": 44100}

        has_a = chunk_audio_a is not None and "waveform" in chunk_audio_a and chunk_audio_a["waveform"].shape[-1] > 0
        has_b = chunk_audio_b is not None and "waveform" in chunk_audio_b and chunk_audio_b["waveform"].shape[-1] > 0
        has_trans = trans_audio is not None and "waveform" in trans_audio and trans_audio["waveform"].shape[-1] > 0

        # Auto-correction for swapped audio tracks based on chunk lengths
        if has_a and has_b:
            sr_test = chunk_audio_a["sample_rate"]
            expected_a_samples = int(round(minimax_temporal_to_frames(a_tokens) * sr_test / fps))
            expected_b_samples = int(round(minimax_temporal_to_frames(b_tokens) * sr_test / fps))
            actual_a_samples = chunk_audio_a["waveform"].shape[-1]
            actual_b_samples = chunk_audio_b["waveform"].shape[-1]

            if abs(actual_a_samples - expected_b_samples) < abs(actual_a_samples - expected_a_samples) and \
               abs(actual_b_samples - expected_a_samples) < abs(actual_b_samples - expected_b_samples):
                print("[LooperMinimaxH3] Detected swapped audio inputs. Auto-correcting chunk_audio_a and chunk_audio_b.")
                chunk_audio_a, chunk_audio_b = chunk_audio_b, chunk_audio_a

        if has_a or has_b or has_trans:
            trimmed_trans_audio = None
            if has_trans:
                sr = trans_audio["sample_rate"]
                w = trans_audio["waveform"]
                trans_total_samples = w.shape[-1]
                overlap_start_samples = int(round(overlap_frames * sr / fps))
                overlap_end_samples = int(round(minimax_temporal_to_frames(end_trim) * sr / fps))
                if trans_total_samples > (overlap_start_samples + overlap_end_samples):
                    trimmed_trans_audio = {
                        "waveform": w[..., overlap_start_samples : trans_total_samples - overlap_end_samples],
                        "sample_rate": sr
                    }
                else:
                    trimmed_trans_audio = trans_audio

            audio_segments = []
            if "A -> B -> Trans" in loop_order:
                if has_a:
                    audio_segments.append(chunk_audio_a)
                if has_b:
                    audio_segments.append(chunk_audio_b)
                if trimmed_trans_audio is not None:
                    audio_segments.append(trimmed_trans_audio)
            else:
                if has_b:
                    audio_segments.append(chunk_audio_b)
                if trimmed_trans_audio is not None:
                    audio_segments.append(trimmed_trans_audio)
                if has_a:
                    audio_segments.append(chunk_audio_a)

            if audio_segments:
                out_audio = concat_audio(audio_segments, crossfade_sec=0.04)

        elapsed = time.perf_counter() - t_start

        # Detailed metrics summary
        print("-" * 65)
        print(f"[LooperMinimaxH3] >>> POSTPROCESS LATENT SUMMARY:")
        print(f"[LooperMinimaxH3]  - Loop Order: {loop_order}")
        if "A -> B -> Trans" in loop_order:
            print(f"[LooperMinimaxH3]  - Sequence: Chunk A ({a_tokens}t) -> Chunk B ({b_tokens}t) -> Trans ({trimmed_tokens}t)")
        else:
            print(f"[LooperMinimaxH3]  - Sequence: Chunk B ({b_tokens}t) -> Trans ({trimmed_tokens}t) -> Chunk A ({a_tokens}t)")
        print(f"[LooperMinimaxH3]  - Guide Overlap Trimmed: {start_trim} tokens start (~{overlap_frames}f), {end_trim} tokens end (~{minimax_temporal_to_frames(end_trim)}f)")
        print(f"[LooperMinimaxH3]  - Assembled Loop Latent: {total_tokens} tokens (~{total_length_frames} frames) in {elapsed:.4f}s")
        print("=" * 65 + "\n")

        return (
            out_latent,
            out_audio,
            total_length_frames
        )
