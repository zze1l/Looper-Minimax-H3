"""
MiniMax H3 Core Integration Module
Encapsulates direct execution of MiniMax H3 reference-to-video and guide injection
from comfy_extras, ensuring decoupled architecture across all node implementations.
Also provides active diffusion model continuity patching (MiniMaxH3ContinuityWrapper)
to lock boundary context guide latents and prevent drift during transition generation.
"""

import math
import torch
from .looper_utils import (
    extract_video_audio_latents,
    minimax_temporal_to_frames,
    minimax_audio_span,
    wrap_pad_periodic,
    MINIMAX_CLIP_LATENTS,
    MINIMAX_VIDEO_TAIL
)


def format_minimax_loop_prompt(
    prompt: str,
    auto_format: bool = True,
    use_dual_pictures: bool = False
) -> str:
    """
    Formats a user transition/loop prompt into the native MiniMax H3 Qwen3-VL multimodal structure
    (inspired by ComfyUI-MiniMaxH3-Loop PROMPTING.md):
    - Ensures explicit `<Picture 1>` visual subject anchor so Qwen3-VL links the reference image
      without treating it as a hard shot cut
    - Appends continuous state reinforcement ('Continuous, smooth, unbroken motion throughout.')
    - Appends `overall_soundscape:` and `non_diegetic_music: N/A` when missing to stabilize
      joint audio-video DiT attention tokens and prevent garbled audio hallucinations.
    """
    clean = (prompt or "").strip()
    if not auto_format:
        return clean

    if not clean:
        base_visual = "The scene in <Picture 1> continues in steady, natural motion."
        if use_dual_pictures:
            base_visual = "The scene in <Picture 1> flows smoothly into <Picture 2> in steady, natural motion."
        return (
            f"{base_visual} Continuous, smooth, unbroken motion throughout.\n\n"
            "overall_soundscape: steady ambient room tone, natural background atmosphere\n\n"
            "non_diegetic_music: N/A"
        )

    # Separate visual prose from existing audio directives if present
    parts = clean.split("overall_soundscape:", 1)
    visual_part = parts[0].strip()
    sound_part = ("overall_soundscape:" + parts[1]) if len(parts) > 1 else ""

    # 1. Inject <Picture 1> subject reference if absent
    if "<Picture 1>" not in visual_part:
        if use_dual_pictures and "<Picture 2>" not in visual_part:
            visual_part = f"In <Picture 1> transitioning smoothly toward <Picture 2>, {visual_part}"
        else:
            visual_part = f"In <Picture 1>, {visual_part}"
    elif use_dual_pictures and "<Picture 2>" not in visual_part:
        visual_part = f"{visual_part} Flowing seamlessly into <Picture 2>."

    # 2. Reinforce continuous cyclic state if not already present
    lower_vis = visual_part.lower()
    if "continuous, smooth, unbroken" not in lower_vis and "unbroken motion" not in lower_vis:
        if not visual_part.endswith((".", "!", "?")):
            visual_part += "."
        visual_part += " Continuous, smooth, unbroken motion throughout."

    # 3. Ensure steady soundscape directives for the AV DiT stream
    if not sound_part:
        sound_part = (
            "overall_soundscape: steady ambient atmosphere, smooth continuous background tone\n\n"
            "non_diegetic_music: N/A"
        )
    elif "non_diegetic_music:" not in sound_part:
        sound_part = sound_part.strip() + "\n\nnon_diegetic_music: N/A"

    return f"{visual_part}\n\n{sound_part.strip()}"


def apply_minimax_h3_guide(positive, latent, vae, image, frame_idx: int, audio=None, audio_vae=None):
    """Applies MiniMaxH3AddGuide directly from comfy_extras with optional audio."""
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
        res = MiniMaxH3AddGuide.execute(
            positive=positive,
            latent=latent,
            frame_idx=frame_idx,
            vae=vae,
            audio_vae=audio_vae,
            image=image,
            audio=audio
        )
        if hasattr(res, "args") and len(res.args) > 0:
            return res.args[0]
        elif hasattr(res, "result") and res.result is not None:
            return res.result[0]
        elif isinstance(res, (tuple, list)):
            return res[0]
        return res
    except Exception as e:
        print(f"[LooperMinimaxH3] Warning applying MiniMaxH3AddGuide (frame_idx={frame_idx}): {e}")
        import traceback
        traceback.print_exc()
    return positive


def apply_minimax_h3_continuation_guides(
    positive,
    trans_empty_latent: dict,
    chunk_latent_b: dict,
    chunk_latent_a: dict,
    vae=None,
    audio_vae=None,
    guide_b_audio=None,
    guide_a_audio=None,
    decoded_guide_b=None,
    decoded_guide_a=None,
    k_tokens: int = 7
):
    """
    Applies pure latent continuation keyframes directly into MiniMax H3 positive conditioning.
    Adopts the community video-continuation paradigm (e.g. ComfyUI-MiniMax-H3-Extend):
    - Extracts raw, uncompressed latent context tokens directly from chunk_latent_b (tail) and chunk_latent_a (head)
    - Supports configurable `k_tokens` (e.g. 7 tokens = 22f, 5 tokens = 17f, 2 tokens = 5f) to balance
      boundary lock vs. prompt freedom and sampling speed
    - Injects them natively into 'minimax_keyframes' without lossy vae.decode -> vae.encode round-trips
    - Anchor 1 (Origin): Chunk B tail at resolved_frame_index = 0
    - Anchor 2 (Destination): Chunk A head at resolved_frame_index = total_frames - guide_frames
    - ComfyUI's PackedLayout positions them on the DiT temporal RoPE grid as conditioning tokens (img_update=False)
    - Preserves unhindered diffusion ODE velocities so the sampler produces a natural, seamless transition bridge
    """
    if positive is None:
        return positive

    k_tokens = max(1, int(k_tokens))
    guide_frames_default = minimax_temporal_to_frames(k_tokens)

    try:
        video_b, audio_b_lat = extract_video_audio_latents(chunk_latent_b)
        video_a, audio_a_lat = extract_video_audio_latents(chunk_latent_a)
        trans_v, trans_a_lat = extract_video_audio_latents(trans_empty_latent)

        if trans_v is None or video_b is None or video_a is None:
            raise ValueError("Invalid latent tensors for continuation keyframing")

        T_trans = trans_v.shape[2]
        total_frames = minimax_temporal_to_frames(T_trans)

        eff_k_b = min(k_tokens, video_b.shape[2])
        eff_k_a = min(k_tokens, video_a.shape[2])
        guide_frames_a = minimax_temporal_to_frames(eff_k_a)

        # Slice pure raw latent tokens (bit-exact from source, zero VAE re-encode loss)
        raw_b_tail = video_b[:, :, -eff_k_b:, :, :].clone()
        raw_a_head = video_a[:, :, :eff_k_a, :, :].clone()

        # Build native keyframes
        keyframe_b = {
            "resolved_frame_index": 0,
            "latent": raw_b_tail
        }

        resolved_a = max(0, total_frames - guide_frames_a)
        keyframe_a = {
            "resolved_frame_index": resolved_a,
            "latent": raw_a_head
        }

        # Attach audio latents if present
        if audio_b_lat is not None and audio_a_lat is not None:
            audio_k = int(round(k_tokens * (40 / 24)))
            if audio_b_lat.shape[-1] >= audio_k:
                keyframe_b["audio_latent"] = audio_b_lat[..., -audio_k:].clone()
            if audio_a_lat.shape[-1] >= audio_k:
                keyframe_a["audio_latent"] = audio_a_lat[..., :audio_k].clone()

        import node_helpers
        if isinstance(positive, list) and len(positive) > 0 and isinstance(positive[0], (tuple, list)) and len(positive[0]) > 1:
            keyframes = list(positive[0][1].get("minimax_keyframes", []))
        else:
            keyframes = []

        keyframes.append(keyframe_b)
        keyframes.append(keyframe_a)

        positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})
        print(f"[LooperMinimaxH3] >>> Injected Native Latent Continuation Keyframes (k_tokens={k_tokens}):")
        print(f"[LooperMinimaxH3]   - Keyframe B (tail context): resolved_frame_index=0, shape={raw_b_tail.shape}")
        print(f"[LooperMinimaxH3]   - Keyframe A (head context): resolved_frame_index={resolved_a}, shape={raw_a_head.shape}")
        return positive
    except Exception as e:
        print(f"[LooperMinimaxH3] Notice: Native latent keyframe injection fell back to AddGuide: {e}")
        if decoded_guide_a is not None and decoded_guide_b is not None:
            positive = apply_minimax_h3_guide(
                positive=positive,
                latent=trans_empty_latent,
                vae=vae,
                image=decoded_guide_a[:guide_frames_default],
                frame_idx=-guide_frames_default,
                audio=guide_a_audio if audio_vae is not None else None,
                audio_vae=audio_vae
            )
            positive = apply_minimax_h3_guide(
                positive=positive,
                latent=trans_empty_latent,
                vae=vae,
                image=decoded_guide_b[-guide_frames_default:],
                frame_idx=0,
                audio=guide_b_audio if audio_vae is not None else None,
                audio_vae=audio_vae
            )
    return positive


def call_minimax_h3_reference(
    clip,
    vae,
    image_0,
    image_1,
    width: int,
    height: int,
    length: int,
    prompt: str = "",
    audio_vae=None,
    ref_audios=None,
    auto_format_prompt: bool = True,
    use_dual_ref: bool = False
):
    """
    Directly executes MiniMaxH3ReferenceToVideo from comfy_extras.
    Optimizations inspired by ComfyUI-MiniMaxH3-Loop:
    - Formats the prompt into MiniMax H3's native `<Picture 1>` + continuous prose + `overall_soundscape:`
      structure so Qwen3-VL strongly binds the prompt to the visual subject
    - Defaults to single `<Picture 1>` subject anchor (`ref_image_0`) unless `<Picture 2>` is explicitly
      in the prompt or `use_dual_ref=True`, cutting Qwen3-VL vision reference token overhead by 50%
      and preventing two-frame pose conflicts while keyframes handle the exact start/end boundaries.
    """
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        has_pic2_in_prompt = "<Picture 2>" in (prompt or "")
        effective_dual = bool((use_dual_ref or has_pic2_in_prompt) and image_1 is not None)

        formatted_prompt = format_minimax_loop_prompt(
            prompt=prompt,
            auto_format=auto_format_prompt,
            use_dual_pictures=effective_dual
        )

        ref_images = {"ref_image_0": image_0}
        if effective_dual:
            ref_images["ref_image_1"] = image_1

        kwargs = {
            "clip": clip,
            "prompt": formatted_prompt,
            "width": width,
            "height": height,
            "length": length,
            "ref_image_size": "match",
            "vae": vae,
            "audio_vae": audio_vae,
            "ref_images": ref_images
        }
        if ref_audios and audio_vae is not None:
            if effective_dual:
                kwargs["ref_audios"] = ref_audios
            elif "ref_audio_0" in ref_audios:
                kwargs["ref_audios"] = {"ref_audio_0": ref_audios["ref_audio_0"]}

        res = MiniMaxH3ReferenceToVideo.execute(**kwargs)
        
        pos = None
        lat = None
        if hasattr(res, "args") and len(res.args) >= 2:
            pos, lat = res.args[0], res.args[1]
        elif hasattr(res, "result") and res.result is not None and len(res.result) >= 2:
            pos, lat = res.result[0], res.result[1]
        elif isinstance(res, (tuple, list)) and len(res) >= 2:
            pos, lat = res[0], res[1]

        if pos is not None and lat is not None:
            print(f"[LooperMinimaxH3] MiniMaxH3ReferenceToVideo executed successfully (ref_anchors={len(ref_images)}, length={length}f)!")
            return pos, lat
    except Exception as e:
        print(f"[LooperMinimaxH3] Error calling MiniMaxH3ReferenceToVideo: {e}")
        import traceback
        traceback.print_exc()
    return None, None


def clamp_minimax_continuity(
    out_video: torch.Tensor,
    target_b_video: torch.Tensor,
    target_a_video: torch.Tensor,
    k_tokens: int = 7,
    blend_tokens: int = 2,
    strength: float = 1.0
) -> torch.Tensor:
    """
    Clamps the predicted denoised latent x0 at the context guide boundaries
    (Chunk B tail at the beginning, Chunk A head at the end) with a smooth cosine ramp.
    Preserves 100% video continuity into and out of the generated transition bridge.
    """
    if strength <= 0.0 or k_tokens <= 0:
        return out_video
    if not isinstance(out_video, torch.Tensor) or out_video.ndim != 5:
        return out_video
    if target_b_video is None or target_a_video is None:
        return out_video
    if not isinstance(target_b_video, torch.Tensor) or not isinstance(target_a_video, torch.Tensor):
        return out_video
    if target_b_video.ndim != 5 or target_a_video.ndim != 5:
        return out_video

    B, C, T, H, W = out_video.shape
    T_b = target_b_video.shape[2]
    T_a = target_a_video.shape[2]
    eff_k = min(k_tokens, T_b, T_a, T // 2)
    if eff_k <= 0:
        return out_video

    # Check spatial dimensions match
    if target_b_video.shape[3:] != out_video.shape[3:] or target_a_video.shape[3:] != out_video.shape[3:]:
        return out_video

    # Extract target boundary tokens
    ref_b = target_b_video[:, :, -eff_k:, :, :].to(device=out_video.device, dtype=out_video.dtype)
    ref_a = target_a_video[:, :, :eff_k, :, :].to(device=out_video.device, dtype=out_video.dtype)

    if ref_b.shape[0] != B:
        ref_b = ref_b.expand(B, -1, -1, -1, -1)
    if ref_a.shape[0] != B:
        ref_a = ref_a.expand(B, -1, -1, -1, -1)

    result = out_video.clone()
    b_blend = max(0, min(blend_tokens, eff_k))
    lock_b = eff_k - b_blend

    # Chunk B tail (beginning of transition)
    for t in range(eff_k):
        if t < lock_b:
            w = 1.0 * strength
        else:
            idx = t - lock_b
            ramp = 0.5 * (1.0 + math.cos(math.pi * (idx + 1) / (b_blend + 1)))
            w = ramp * strength
        result[:, :, t, :, :] = (1.0 - w) * result[:, :, t, :, :] + w * ref_b[:, :, t, :, :]

    # Chunk A head (end of transition)
    for t in range(T - eff_k, T):
        rel_t = t - (T - eff_k)
        if rel_t < b_blend:
            idx = b_blend - 1 - rel_t
            ramp = 0.5 * (1.0 + math.cos(math.pi * (idx + 1) / (b_blend + 1)))
            w = ramp * strength
        else:
            w = 1.0 * strength
        result[:, :, t, :, :] = (1.0 - w) * result[:, :, t, :, :] + w * ref_a[:, :, rel_t, :, :]

    return result


def clamp_minimax_audio_continuity(
    out_audio: torch.Tensor,
    target_b_audio: torch.Tensor,
    target_a_audio: torch.Tensor,
    k_audio_tokens: int = 12,
    blend_tokens: int = 3,
    strength: float = 1.0
) -> torch.Tensor:
    """Clamps audio latents at boundary context tokens."""
    if strength <= 0.0 or k_audio_tokens <= 0:
        return out_audio
    if not isinstance(out_audio, torch.Tensor) or out_audio.ndim < 3:
        return out_audio
    if target_b_audio is None or target_a_audio is None:
        return out_audio
    if not isinstance(target_b_audio, torch.Tensor) or not isinstance(target_a_audio, torch.Tensor):
        return out_audio

    T_out = out_audio.shape[-1]
    T_b = target_b_audio.shape[-1]
    T_a = target_a_audio.shape[-1]
    eff_k = min(k_audio_tokens, T_b, T_a, T_out // 2)
    if eff_k <= 0:
        return out_audio

    ref_b = target_b_audio[..., -eff_k:].to(device=out_audio.device, dtype=out_audio.dtype)
    ref_a = target_a_audio[..., :eff_k].to(device=out_audio.device, dtype=out_audio.dtype)

    result = out_audio.clone()
    b_blend = max(0, min(blend_tokens, eff_k))
    lock_b = eff_k - b_blend

    for t in range(eff_k):
        if t < lock_b:
            w = 1.0 * strength
        else:
            idx = t - lock_b
            ramp = 0.5 * (1.0 + math.cos(math.pi * (idx + 1) / (b_blend + 1)))
            w = ramp * strength
        result[..., t] = (1.0 - w) * result[..., t] + w * ref_b[..., t]

    for t in range(T_out - eff_k, T_out):
        rel_t = t - (T_out - eff_k)
        if rel_t < b_blend:
            idx = b_blend - 1 - rel_t
            ramp = 0.5 * (1.0 + math.cos(math.pi * (idx + 1) / (b_blend + 1)))
            w = ramp * strength
        else:
            w = 1.0 * strength
        result[..., t] = (1.0 - w) * result[..., t] + w * ref_a[..., rel_t]

    return result


def minimax_loop_sampler_wrapper(executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs):
    """
    ComfyUI WrappersMP.SAMPLER_SAMPLE wrapper (inspired by ComfyUI-MiniMaxH3-Loop):
    Generates a fresh orthogonal noise swap tensor for wrapped circular padding regions so
    distilled 8-step models (FastH3-Omni / Turbo LoRA) never see identical noise twice near the loop seam.
    """
    try:
        import comfy.utils
        shapes = getattr(guider.inner_model, "latent_shapes", None)
        video_noise = comfy.utils.unpack_latents(noise, shapes)[0] if shapes else noise
        if torch.count_nonzero(video_noise) == 0:
            swap = video_noise
        else:
            seed = ((extra_args.get("seed") or 0) + 1) % (2 ** 64)
            generator = torch.Generator(device=video_noise.device).manual_seed(seed)
            swap = torch.randn(
                video_noise.shape,
                generator=generator,
                device=video_noise.device,
                dtype=video_noise.dtype
            ) - video_noise
        extra_args.setdefault("model_options", {}).setdefault("transformer_options", {})[
            "minimax_h3_loop_noise_swap"
        ] = swap
    except Exception as e:
        print(f"[LooperMinimaxH3] Warning in minimax_loop_sampler_wrapper: {e}")
    return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)


def minimax_circular_diffusion_wrapper(executor, x, timestep, context, transformer_options, **kwargs):
    """
    ComfyUI WrappersMP.DIFFUSION_MODEL wrapper (inspired by ComfyUI-MiniMaxH3-Loop):
    Wraps `pad_clips` (5 tokens = 17 frames) around both sides of the video & audio latent at each
    diffusion step and injects sigma-scaled fresh noise (`torch.addcmul(..., sigma, swap)`),
    preserving 100% Flow-Matching ODE velocity consistency across the circular boundary.
    """
    pad = int(transformer_options.get("minimax_h3_loop_pad_clips", 0))
    if pad <= 0 or not isinstance(x, (list, tuple)) or len(x) < 2:
        return executor(x, timestep, context, transformer_options, **kwargs)

    video, audio = x[0], x[1]
    n = video.shape[2] - MINIMAX_VIDEO_TAIL
    if n <= 0 or n % MINIMAX_CLIP_LATENTS != 0:
        return executor(x, timestep, context, transformer_options, **kwargs)

    clips = n // MINIMAX_CLIP_LATENTS
    na = minimax_audio_span(clips)
    audio_tail = audio.shape[-1] - na
    pv = pad * MINIMAX_CLIP_LATENTS
    pa = minimax_audio_span(pad)

    args = dict(kwargs)
    payload = kwargs.get("minimax_payload")
    if payload and payload.get("keyframes"):
        frames = clips * 17
        keyframes = [
            dict(kf, resolved_frame_index=pad * 17 + (kf["resolved_frame_index"] % max(1, frames)))
            for kf in payload["keyframes"]
        ]
        args["minimax_payload"] = dict(payload, keyframes=keyframes)

    if kwargs.get("denoise_mask") is not None:
        args["denoise_mask"] = wrap_pad_periodic(kwargs["denoise_mask"][:, :, :n], pv, pv + MINIMAX_VIDEO_TAIL, 2)
    if kwargs.get("audio_denoise_mask") is not None:
        args["audio_denoise_mask"] = wrap_pad_periodic(kwargs["audio_denoise_mask"][..., :na], pa, pa + audio_tail, -1)

    padded_video = wrap_pad_periodic(video[:, :, :n], pv, pv + MINIMAX_VIDEO_TAIL, 2)
    noise_swap = transformer_options.get("minimax_h3_loop_noise_swap")
    sigmas = transformer_options.get("sigmas")
    if noise_swap is not None and sigmas is not None and noise_swap.shape[2] >= n:
        rep = max(1, video.shape[0] // max(1, noise_swap.shape[0]))
        ns = noise_swap.repeat(rep, 1, 1, 1, 1)
        swap = wrap_pad_periodic(ns[:, :, :n], pv, pv + MINIMAX_VIDEO_TAIL, 2).to(video.dtype)
        swap[:, :, pv:pv + n] = 0
        if args.get("denoise_mask") is not None:
            swap = swap * args["denoise_mask"]
        sigma = sigmas.view(-1, 1, 1, 1, 1).to(video.dtype)
        padded_video = torch.addcmul(padded_video, sigma, swap)

    padded_audio = wrap_pad_periodic(audio[..., :na], pa, pa + audio_tail, -1)
    out = executor([padded_video, padded_audio], timestep, context, transformer_options, **args)
    v_out = out[0][:, :, pv:pv + n]
    a_out = out[1][..., pa:pa + na]
    return [wrap_pad_periodic(v_out, 0, MINIMAX_VIDEO_TAIL, 2), wrap_pad_periodic(a_out, 0, audio_tail, -1)]


class MiniMaxH3ContinuityWrapper:
    """
    ComfyUI WrappersMP.APPLY_MODEL wrapper for MiniMax H3.
    Intercepts the predicted denoised latent x0 at each step and clamps
    the context guide boundaries to Chunk B tail and Chunk A head.
    Supports optional Flow-Matching sigma scheduling (`sigma_schedule=True`) so boundary clamping
    anchors the coarse trajectory at high sigma and smoothly releases at low sigma for crisp detail.
    """
    def __init__(self, target_b_video, target_a_video, k_tokens=7, blend_tokens=2, strength=1.0,
                 target_b_audio=None, target_a_audio=None, sigma_schedule=False):
        self.target_b_video = target_b_video
        self.target_a_video = target_a_video
        self.k_tokens = k_tokens
        self.blend_tokens = blend_tokens
        self.strength = strength
        self.target_b_audio = target_b_audio
        self.target_a_audio = target_a_audio
        self.sigma_schedule = sigma_schedule

    def _effective_strength(self, t, transformer_options: dict) -> float:
        if not self.sigma_schedule:
            return float(self.strength)
        try:
            sigmas = transformer_options.get("sigmas") if isinstance(transformer_options, dict) else None
            sigma_val = None
            if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
                sigma_val = float(sigmas.flatten()[0].item())
            elif isinstance(t, torch.Tensor) and t.numel() > 0:
                val = float(t.flatten()[0].item())
                sigma_val = val / 1000.0 if val > 1.0 else val
            if sigma_val is not None:
                # Smooth Flow-Matching schedule: full anchor for sigma >= 0.3, cosine release as sigma -> 0
                s_clamped = max(0.0, min(1.0, sigma_val))
                if s_clamped < 0.3:
                    gate = 0.5 * (1.0 - math.cos(math.pi * (s_clamped / 0.3)))
                    return float(self.strength) * gate
        except Exception:
            pass
        return float(self.strength)

    def __call__(self, executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        out = executor(x, t, c_concat=c_concat, c_crossattn=c_crossattn, control=control, transformer_options=transformer_options, **kwargs)

        eff_strength = self._effective_strength(t, transformer_options)
        if eff_strength <= 0.0 or self.k_tokens <= 0:
            return out

        try:
            if isinstance(out, torch.Tensor) and out.ndim == 5:
                out = clamp_minimax_continuity(
                    out,
                    self.target_b_video,
                    self.target_a_video,
                    k_tokens=self.k_tokens,
                    blend_tokens=self.blend_tokens,
                    strength=eff_strength
                )
            elif isinstance(out, torch.Tensor) and out.ndim == 3 and out.shape[1] == 1:
                # Packed latents (video + audio)
                latent_shapes = transformer_options.get("latent_shapes") or kwargs.get("latent_shapes")
                if latent_shapes is not None:
                    try:
                        import comfy.utils as utils
                        unpacked = utils.unpack_latents(out, latent_shapes)
                        if len(unpacked) >= 1 and unpacked[0].ndim == 5:
                            unpacked[0] = clamp_minimax_continuity(
                                unpacked[0],
                                self.target_b_video,
                                self.target_a_video,
                                k_tokens=self.k_tokens,
                                blend_tokens=self.blend_tokens,
                                strength=eff_strength
                            )
                        if len(unpacked) >= 2 and self.target_b_audio is not None and self.target_a_audio is not None:
                            k_audio = int(round(self.k_tokens * (40 / 24)))
                            unpacked[1] = clamp_minimax_audio_continuity(
                                unpacked[1],
                                self.target_b_audio,
                                self.target_a_audio,
                                k_audio_tokens=k_audio,
                                blend_tokens=max(1, int(round(self.blend_tokens * (40 / 24)))),
                                strength=eff_strength
                            )
                        out, _ = utils.pack_latents(unpacked)
                    except Exception as e:
                        print(f"[LooperMinimaxH3] Warning in continuity wrapper unpacking: {e}")
        except Exception as e:
            print(f"[LooperMinimaxH3] Warning in MiniMaxH3ContinuityWrapper: {e}")

        return out


def patch_minimax_h3_continuity(
    model,
    chunk_latent_b: dict = None,
    chunk_latent_a: dict = None,
    k_tokens: int = 7,
    blend_tokens: int = 2,
    strength: float = 0.0,
    pad_clips: int = 0,
    sigma_schedule: bool = False
):
    """
    Clones the ComfyUI ModelPatcher and installs:
    - MiniMaxH3ContinuityWrapper onto WrappersMP.APPLY_MODEL (when strength > 0.0 and chunks provided)
    - Circular periodic diffusion padding (`minimax_circular_diffusion_wrapper` + `minimax_loop_sampler_wrapper`)
      when `pad_clips > 0` (inspired by ComfyUI-MiniMaxH3-Loop).
    """
    if model is None:
        return None

    if strength <= 0.0 and pad_clips <= 0:
        print("[LooperMinimaxH3] Continuity Patcher: Operating in Passthrough mode (full ODE flow velocity preserved).")
        return model

    # Clone the model patcher to avoid mutating the base model
    patched_model = model.clone() if hasattr(model, "clone") else model

    if pad_clips > 0:
        try:
            import comfy.patcher_extension
            if hasattr(patched_model, "model_options"):
                t_opts = patched_model.model_options.setdefault("transformer_options", {})
                t_opts["minimax_h3_loop_pad_clips"] = int(pad_clips)
                t_opts["minimax_h3_loop_views"] = 1
            if hasattr(patched_model, "add_wrapper_with_key"):
                patched_model.add_wrapper_with_key(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    "minimax_h3_loop",
                    minimax_circular_diffusion_wrapper
                )
                patched_model.add_wrapper_with_key(
                    comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
                    "minimax_h3_loop",
                    minimax_loop_sampler_wrapper
                )
            elif hasattr(patched_model, "add_wrapper"):
                patched_model.add_wrapper(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    minimax_circular_diffusion_wrapper
                )
            print(f"[LooperMinimaxH3] Installed Circular Periodic Diffusion Wrapper (pad_clips={pad_clips}, views=1)")
        except Exception as e:
            print(f"[LooperMinimaxH3] Warning installing circular diffusion wrapper: {e}")

    if strength > 0.0 and chunk_latent_b is not None and chunk_latent_a is not None:
        target_b_video, target_b_audio = extract_video_audio_latents(chunk_latent_b)
        target_a_video, target_a_audio = extract_video_audio_latents(chunk_latent_a)

        wrapper = MiniMaxH3ContinuityWrapper(
            target_b_video=target_b_video,
            target_a_video=target_a_video,
            k_tokens=k_tokens,
            blend_tokens=blend_tokens,
            strength=strength,
            target_b_audio=target_b_audio,
            target_a_audio=target_a_audio,
            sigma_schedule=sigma_schedule
        )

        try:
            import comfy.patcher_extension
            patched_model.add_wrapper(comfy.patcher_extension.WrappersMP.APPLY_MODEL, wrapper)
            print(f"[LooperMinimaxH3] Installed MiniMaxH3ContinuityWrapper on model (k={k_tokens}, blend={blend_tokens}, strength={strength:.2f})")
        except Exception as e:
            if hasattr(patched_model, "add_wrapper"):
                try:
                    patched_model.add_wrapper("apply_model", wrapper)
                    print(f"[LooperMinimaxH3] Installed wrapper via string key on model.")
                except Exception as e2:
                    print(f"[LooperMinimaxH3] Error registering wrapper on model: {e2}")
            else:
                print(f"[LooperMinimaxH3] Error registering wrapper on model: {e}")

    return patched_model
