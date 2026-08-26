"""diffusion.py — flow-matching mechanics for pose-controlnet: patchify,
resolution-aware timestep schedule, control-concat forward pass, and a short
euler sampler for periodic eval images.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base_model"))
from mmdit import _mask, temb  # noqa: E402


def patchify_and_position(latent: torch.Tensor, txt_len: int, patch: int,
                           txt_mask: torch.Tensor):
    b, _, h, w = latent.shape
    h_, w_ = h // patch, w // patch

    ids = torch.zeros((h_, w_, 3), device=latent.device)
    ids[..., 1] = torch.arange(h_, device=latent.device)[:, None]
    ids[..., 2] = torch.arange(w_, device=latent.device)[None, :]
    img_pos = repeat(ids, "h w c -> b (h w) c", b=b, c=3)
    img_mask = torch.ones(b, h_ * w_, device=latent.device, dtype=torch.bool)

    tokens = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txt_pos = torch.zeros(b, txt_len, 3, device=latent.device)
    pos = torch.cat((txt_pos, img_pos), dim=1)
    mask = torch.cat((txt_mask, img_mask), dim=1)
    return tokens, pos, mask


def resolution_shift_mu(seq_len: int, x1: float, y1: float, x2: float, y2: float) -> float:
    slope = (y2 - y1) / (x2 - x1)
    return slope * seq_len + (y1 - slope * x1)


def shift_timestep(t: torch.Tensor, mu: float) -> torch.Tensor:
    return math.exp(mu) * t / (math.exp(mu) * t + 1.0 - t)


def sample_flow_timestep(batch_size: int, seq_len: int, cfg, device,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample the intended logistic-normal timestep and apply resolution shift."""
    if batch_size < 1 or seq_len < 1:
        raise ValueError("batch_size and seq_len must be positive")
    mu = resolution_shift_mu(seq_len, cfg.mu_x1, cfg.mu_y1, cfg.mu_x2, cfg.mu_y2)
    logistic_normal = torch.sigmoid(
        torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    )
    shifted = shift_timestep(logistic_normal, mu)
    if not torch.isfinite(shifted).all().item() or not ((shifted > 0) & (shifted < 1)).all().item():
        raise FloatingPointError("Sampled invalid flow-matching timestep")
    return shifted


def make_flow_pair(clean_image: torch.Tensor, noise: torch.Tensor,
                   timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct x_t and v target while leaving any control latent untouched."""
    if clean_image.shape != noise.shape:
        raise ValueError("clean image latent and noise must have identical shapes")
    if timestep.ndim != 1 or timestep.shape[0] != clean_image.shape[0]:
        raise ValueError("timestep must have shape (batch,)")
    broadcast = timestep.view(-1, *([1] * (clean_image.ndim - 1)))
    noisy = broadcast * noise + (1.0 - broadcast) * clean_image
    target = noise - clean_image
    if not torch.isfinite(noisy).all().item() or not torch.isfinite(target).all().item():
        raise FloatingPointError("Non-finite flow-matching input or target")
    return noisy, target


def flow_schedule(seq_len: int, steps: int, cfg) -> list[float]:
    mu = resolution_shift_mu(seq_len, cfg.mu_x1, cfg.mu_y1, cfg.mu_x2, cfg.mu_y2)
    ts = torch.linspace(1, 0, steps + 1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0))
    return ts.tolist()


def checkpointed_main_block_indices(block_count: int, requested_blocks: int) -> range:
    """Return the contiguous prefix of main blocks selected for recomputation.

    The first ``requested_blocks`` entries of ``model.blocks`` are checkpointed
    in execution order.  A prefix keeps tuning deterministic and makes zero
    mean no checkpointing.  The caller supplies the actual model block count,
    so this rejects requests incompatible with a loaded checkpoint.
    """
    if isinstance(requested_blocks, bool) or not isinstance(requested_blocks, int):
        raise TypeError("gradient checkpointing block count must be an integer")
    if not 0 <= requested_blocks <= block_count:
        raise ValueError(
            f"gradient checkpointing block count must be in [0, {block_count}], got {requested_blocks}"
        )
    return range(requested_blocks)


def forward_pose_control(model, noisy_img, pose_ctrl, context, t, pos, mask,
                         gradient_checkpointing_blocks: int | None = None,
                         grad_ckpt: bool | None = True):
    """Run Krea with optional checkpointing of a prefix of its main blocks.

    ``grad_ckpt`` remains a compatibility alias: ``True`` checkpoints every
    main block and ``False`` checkpoints none.  New callers must provide
    ``gradient_checkpointing_blocks`` (0 through ``len(model.blocks)``).
    """
    if gradient_checkpointing_blocks is None:
        gradient_checkpointing_blocks = len(model.blocks) if grad_ckpt else 0
    checkpointed_indices = checkpointed_main_block_indices(
        len(model.blocks), gradient_checkpointing_blocks
    )
    x = model.first(torch.cat([noisy_img, pose_ctrl], dim=-1))
    t_raw = model.tmlp(temb(t, model.config.tdim, device=x.device, dtype=x.dtype))
    t_vec = model.tproj(t_raw)

    with torch.no_grad():
        txt_mask = _mask(mask[:, : context.shape[1]])
        context = model.txtfusion(context, mask=txt_mask)
        context = model.txtmlp(context)

    txt_len, img_len = context.shape[1], x.shape[1]
    combined = torch.cat((context, x), dim=1)

    pad = (-combined.shape[1]) % 256
    if pad > 0:
        combined = F.pad(combined, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad), value=False)
        pos = F.pad(pos, (0, 0, 0, pad))

    full_mask = _mask(mask)
    freqs = model.posemb(pos)

    for index, block in enumerate(model.blocks):
        if index in checkpointed_indices:
            combined = checkpoint(block, combined, t_vec, freqs, full_mask, use_reentrant=False)
        else:
            combined = block(combined, t_vec, freqs, full_mask)

    out = model.last(combined, t_raw)
    return out[:, txt_len: txt_len + img_len, :]


@torch.no_grad()
def sample_eval_image(model, vae_decode_fn, conditioner, sample, cfg, device, seed: int):
    latent = sample["latent"][None].to(device)
    ctrl_latent = sample["control"][None].to(device, torch.bfloat16)
    patch = model.config.patch

    noise = torch.randn(latent.shape, device=device, dtype=torch.bfloat16,
                        generator=torch.Generator(device=device).manual_seed(seed))
    if conditioner is None:
        required = ("context", "mask", "unconditional_context", "unconditional_mask")
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"Cached evaluation conditioning is incomplete: missing={missing}")
        txt, txt_mask = sample["context"][None].to(device, torch.bfloat16), sample["mask"][None].to(device, torch.bool)
        untxt, untxt_mask = (sample["unconditional_context"][None].to(device, torch.bfloat16),
                             sample["unconditional_mask"][None].to(device, torch.bool))
    else:
        txt, txt_mask = conditioner([sample["prompt"]])
        untxt, untxt_mask = conditioner([""])

    img, pos, mask = patchify_and_position(noise, txt.shape[1], patch, txt_mask)
    _, unpos, unmask = patchify_and_position(noise, untxt.shape[1], patch, untxt_mask)
    ctrl, _, _ = patchify_and_position(ctrl_latent, txt.shape[1], patch, txt_mask)

    ts = flow_schedule(img.shape[1], cfg.eval_steps, cfg)
    for t_curr, t_prev in zip(ts[:-1], ts[1:]):
        t = torch.full((1,), t_curr, dtype=img.dtype, device=device)
        cond = forward_pose_control(model, img, ctrl, txt, t, pos, mask, grad_ckpt=False)
        uncond = forward_pose_control(model, img, ctrl, untxt, t, unpos, unmask, grad_ckpt=False)
        v = cond + cfg.eval_guidance * (cond - uncond)
        img = img + (t_prev - t_curr) * v

    h, w = latent.shape[-2:]
    img = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    ph=patch, pw=patch, h=h // patch, w=w // patch)
    pixels = vae_decode_fn(img.to(torch.bfloat16))
    pixels = (pixels.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    return pixels[0].permute(1, 2, 0).float().cpu().byte().numpy()
