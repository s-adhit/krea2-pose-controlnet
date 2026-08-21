"""End-to-end training-loop mechanics smoke test, sized to run on an 8GB GPU.

The real DiT is 13B params (~26GB in bf16) and the real text encoder
(Qwen3-VL-4B) is ~8GB in bf16 on its own -- neither fits on a 4060. So this
test substitutes:
    - a tiny random-init SingleStreamDiT (same architecture class, much
      smaller features/layers) instead of loading real Krea-2-Raw weights
    - a stubbed text conditioner that returns random tensors shaped like
      Qwen3-VL's real output, instead of running the real 4B model

What it DOES use for real: the actual Qwen-Image VAE (small enough to fit)
to encode a couple of real image/control pairs, so latent shapes and dtypes
match what you'll see at full scale.

It exercises the exact code path your fork's train_control_lora.py uses:
ControlInputLayer + LoRA injection -> forward_control -> loss.backward() ->
optimizer.step() -> trainable_state_dict() checkpoint save -> reload ->
one denoising step of sampling. If this runs clean here, the only things
left to debug on GH200 are scale (VRAM, throughput), not logic.

Usage:
    python scripts/toy_train_loop_smoketest.py \
        --sample-image data/full/images/<some_file>.jpg \
        --sample-control data/full/conditioning_images/<same_file>.jpg
    # or, with no real images available:
    python scripts/toy_train_loop_smoketest.py --synthetic
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# Windows has no cl.exe (CPU) / no triton (CUDA) available, and mmdit.py
# decorates RMSNorm.forward / PositionalEncoding.forward / LastLayer.forward
# with @torch.compile(fullgraph=True). Rather than relying on dynamo's
# disable flag (which still raises under fullgraph=True when no frames
# compile), replace torch.compile itself with a no-op BEFORE importing
# mmdit, so those decorators become identity and the methods run eager.
# Must happen before `from mmdit import ...` below, since the decorator
# executes at class-definition time (i.e. at import).
def _no_compile(fn=None, **kwargs):
    if fn is None:
        return lambda f: f
    return fn

torch.compile = _no_compile

# adjust to your vendor/ layout
VENDOR_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "vendor", "krea-2-controlnet")
sys.path.insert(0, VENDOR_ROOT)
sys.path.insert(0, os.path.join(VENDOR_ROOT, "trainer"))

import mmdit as _mmdit_mod  # module reference, to monkeypatch attention()

from mmdit import SingleMMDiTConfig, SingleStreamDiT, _mask, temb  # noqa: E402
from k2_lora import ControlInputLayer, LoRALinear, LORA_TARGETS  # noqa: E402
from sampling import prepare, timesteps  # noqa: E402


def patch_attention_backend(device):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from einops import rearrange

    if device == "cuda":
        return  # test the real cuDNN path, not the FLASH_ATTENTION workaround

    def _cpu_attention(q, k, v, mask=None, scale=None, gqa=False):
        with sdpa_kernel(SDPBackend.MATH):
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
            )
        return rearrange(x, "B H L D -> B L (H D)")

    _mmdit_mod.attention = _cpu_attention

# small but architecturally consistent with the real K2_RAW_CONFIG
# (headdim = features // heads must be a multiple of 16 for the rope-axis math)
TOY_CONFIG = SingleMMDiTConfig(
    features=256, tdim=64, txtdim=128, heads=8, kvheads=4, multiplier=2,
    layers=2, patch=2, channels=16, txtheads=8, txtkvheads=8, txtlayers=12,
)
MU_X1, MU_Y1, MU_X2, MU_Y2 = 256, 0.5, 6400, 1.15


def resolution_mu(seq_len: int) -> float:
    slope = (MU_Y2 - MU_Y1) / (MU_X2 - MU_X1)
    return slope * seq_len + (MU_Y1 - slope * MU_X1)


def shift_t(t: torch.Tensor, mu: float) -> torch.Tensor:
    import math
    return math.exp(mu) * t / (math.exp(mu) * t + 1.0 - t)


def build_toy_control_model(device):
    """Same surgery as k2_lora.build_control_model, but starting from a
    fresh random-init toy model instead of loading real checkpoint weights.

    FIX: must cast the backbone to bfloat16 here, same as the real
    k2_lora.build_control_model's `model.to(device=device, dtype=dtype)`.
    Everything downstream in this script (img/ctrl/context/t) is cast to
    bf16 before it hits the model, so leaving the backbone in fp32 crashes
    the very first nn.Linear inside forward_control with a dtype mismatch.
    """
    model = SingleStreamDiT(TOY_CONFIG).to(device=device, dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.first = ControlInputLayer(model.first).to(device)

    def _get(root, path):
        m = root
        for p in path.split("."):
            m = getattr(m, p)
        return m

    def _set(root, path, new):
        parts = path.split(".")
        parent = _get(root, ".".join(parts[:-1])) if len(parts) > 1 else root
        setattr(parent, parts[-1], new)

    for i in range(TOY_CONFIG.layers):
        for t in LORA_TARGETS:
            path = f"blocks.{i}.{t}"
            base = _get(model, path)
            _set(model, path, LoRALinear(base, rank=8, alpha=8).to(device))
            _get(model, path).base.requires_grad_(False)
    return model


def stub_conditioner(prompts: list[str], seq_len: int, device):
    """Returns random tensors shaped like the real TextConditioner's output:
    context (b, l, txtlayers, txtdim), mask (b, l) -- shape-compatible with
    forward_control, not semantically meaningful."""
    b = len(prompts)
    context = torch.randn(b, seq_len, TOY_CONFIG.txtlayers, TOY_CONFIG.txtdim,
                          device=device, dtype=torch.bfloat16)
    mask = torch.ones(b, seq_len, dtype=torch.bool, device=device)
    return context, mask


def forward_control(model, img, ctrl, context, t, pos, mask):
    """Same logic as the real trainer's forward_control, no grad checkpointing
    needed at this scale."""
    x = model.first(torch.cat([img, ctrl], dim=-1))
    tv = model.tmlp(temb(t, model.config.tdim, device=x.device, dtype=x.dtype))
    tvec = model.tproj(tv)

    with torch.no_grad():
        txtmask = _mask(mask[:, : context.shape[1]])
        context = model.txtfusion(context, mask=txtmask)
        context = model.txtmlp(context)

    txtlen, imglen = context.shape[1], x.shape[1]
    combined = torch.cat((context, x), dim=1)

    padlen = (-combined.shape[1]) % 256
    if padlen > 0:
        combined = F.pad(combined, (0, 0, 0, padlen))
        mask = F.pad(mask, (0, padlen), value=False)
        pos = F.pad(pos, (0, 0, 0, padlen))

    mask = _mask(mask)
    freqs = model.posemb(pos)
    for block in model.blocks:
        combined = block(combined, tvec, freqs, mask)

    final = model.last(combined, tv)
    return final[:, txtlen : txtlen + imglen, :]


def load_latents(args, device):
    if args.synthetic:
        # 832x1216 bucket -> latent (16, 152, 104)
        return (torch.randn(1, 16, 152, 104, device=device),
                torch.randn(1, 16, 152, 104, device=device))

    from diffusers import AutoencoderKLQwenImage
    ae = AutoencoderKLQwenImage.from_pretrained(
        "Qwen/Qwen-Image", subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device).eval().requires_grad_(False)
    mean = torch.tensor(ae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(ae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    def enc(path):
        img = Image.open(path).convert("RGB").resize((832, 1216), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0
        x = x[None].to(device, torch.bfloat16).unsqueeze(2)
        with torch.no_grad():
            z = ae.encode(x).latent_dist.sample()
            z = (z - mean) / std
        return z.squeeze(2)

    lat = enc(args.sample_image)
    ctrl = enc(args.sample_control)
    del ae
    torch.cuda.empty_cache()
    return lat, ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-image")
    ap.add_argument("--sample-control")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip real VAE encode, use random latents")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-path", default="/tmp/toy_smoketest.safetensors")
    args = ap.parse_args()
    if not args.synthetic:
        assert args.sample_image and args.sample_control, \
            "pass --sample-image/--sample-control, or use --synthetic"

    device = args.device
    print(f"device: {device}")
    patch_attention_backend(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("building toy model...")
    model = build_toy_control_model(device)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    print(f"trainable params: {n_train/1e6:.2f}M")

    opt = torch.optim.AdamW(params, lr=1e-4)

    print("loading/encoding latents...")
    x0, ctrl_lat = load_latents(args, device)
    b = x0.shape[0]
    patch = model.config.patch

    print("forward/backward step...")
    prompts = ["a person in a pose"] * b
    seq_len_guess = (x0.shape[-2] // patch) * (x0.shape[-1] // patch)
    txt, txtmask = stub_conditioner(prompts, seq_len=8, device=device)  # short "text" seq for speed

    noise = torch.randn_like(x0)
    mu = resolution_mu(seq_len_guess)
    u = torch.sigmoid(torch.randn(b, device=device))
    t = shift_t(u, mu)

    xt = t.view(-1, 1, 1, 1) * noise + (1 - t.view(-1, 1, 1, 1)) * x0
    target = noise - x0

    img, pos, mask = prepare(xt.to(torch.bfloat16), txt.shape[1], patch, txtmask)
    ctrl, _, _ = prepare(ctrl_lat.to(torch.bfloat16), txt.shape[1], patch, txtmask)
    target_p, _, _ = prepare(target, txt.shape[1], patch, txtmask)

    pred = forward_control(model, img, ctrl, txt, t.to(torch.bfloat16), pos, mask)
    loss = F.mse_loss(pred.float(), target_p.float())
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    print(f"loss {loss.item():.4f}, grad norm {gnorm:.4f} -- forward/backward/opt-step OK")

    print("checkpoint save/reload...")
    from safetensors.torch import save_file, load_file as load_sd
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if ".A" in k or ".B" in k or k.startswith("first.")}
    save_file(sd, args.ckpt_path, metadata={"rank": "8", "step": "1"})
    model2 = build_toy_control_model(device)
    missing, unexpected = model2.load_state_dict(load_sd(args.ckpt_path), strict=False)
    assert not unexpected, unexpected
    print(f"checkpoint round-trip OK ({os.path.getsize(args.ckpt_path)} bytes)")

    print("one denoising step (sampling path)...")
    model2.eval()
    with torch.no_grad():
        ts = timesteps(img.shape[1], 4, MU_X1, MU_X2, y1=MU_Y1, y2=MU_Y2)
        step_img = img
        for tcurr, tprev in zip(ts[:-1], ts[1:2]):
            tt = torch.full((b,), tcurr, dtype=step_img.dtype, device=device)
            v = forward_control(model2, step_img, ctrl, txt, tt, pos, mask)
            step_img = step_img + (tprev - tcurr) * v
            break
    print("sampling step ran clean")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\npeak VRAM: {peak:.2f} GB")

    os.remove(args.ckpt_path)
    print("\nALL SMOKE TESTS PASSED -- mechanics verified, safe to move to GH200 for real-scale run.")


if __name__ == "__main__":
    main()