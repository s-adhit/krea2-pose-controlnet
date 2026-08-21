"""Does the Qwen-Image VAE preserve thin skeleton lines through an 8x encode/decode
round-trip? This is the highest-priority local smoke test: the control signal your
LoRA trains on is whatever survives this compression, not the original raster.

Runs fine on a RTX 4060 8GB (the VAE alone is far smaller than the 13B DiT) or CPU
(slower, but this only processes ~30-50 small images).

Usage:
    python scripts/vae_roundtrip_sparsity_test.py \
        --diagnostic-manifest data/manifests/diagnostic_val.jsonl \
        --data-root data/full --out-dir data/review/vae_roundtrip \
        --device cuda
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

torch.backends.cudnn.enabled = False
def to_tensor(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32))
    return x.permute(2, 0, 1) / 127.5 - 1.0


def from_tensor(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) * 0.5 + 0.5) * 255.0
    arr = x.permute(1, 2, 0).byte().cpu().numpy()
    return Image.fromarray(arr)


def nonzero_fraction(img: Image.Image, thresh: int = 10) -> float:
    g = np.asarray(img.convert("L"))
    return float((g > thresh).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic-manifest", default="data/manifests/diagnostic_val.jsonl")
    ap.add_argument("--data-root", default="data/full")
    ap.add_argument("--out-dir", default="data/review/vae_roundtrip")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-side", type=int, default=1024,
                    help="resize before VAE encode -- match your bucket scale, not full-res")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from diffusers import AutoencoderKLQwenImage

    print("loading Qwen-Image VAE...")
    ae = AutoencoderKLQwenImage.from_pretrained(
        "Qwen/Qwen-Image", subfolder="vae", torch_dtype=torch.bfloat16
    ).to(args.device).eval().requires_grad_(False)
    mean = torch.tensor(ae.config.latents_mean, device=args.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(ae.config.latents_std, device=args.device).view(1, -1, 1, 1, 1)

    stems = []
    with open(args.diagnostic_manifest) as f:
        for line in f:
            rec = json.loads(line)
            stems.append(rec["file_name"])

    results = []
    for fname in stems:
        stem = os.path.splitext(fname)[0]
        ctrl_path = os.path.join(args.data_root, "conditioning_images", stem + ".png")
        img = Image.open(ctrl_path).convert("RGB")
        w, h = img.size
        scale = args.max_side / max(w, h)
        nw, nh = int(round(w * scale / 16) * 16), int(round(h * scale / 16) * 16)
        img_r = img.resize((nw, nh), Image.LANCZOS)

        x = to_tensor(img_r)[None].to(args.device, torch.bfloat16).unsqueeze(2)
        with torch.no_grad():
            z = ae.encode(x).latent_dist.sample()
            z_norm = (z - mean) / std
            z_denorm = (z_norm * std + mean).to(torch.bfloat16)
            recon = ae.decode(z_denorm).sample

        recon_img = from_tensor(recon[0, :, 0].float())

        orig_frac = nonzero_fraction(img_r)
        recon_frac = nonzero_fraction(recon_img)
        orig_arr = np.asarray(img_r.convert("L"), dtype=np.float32)
        recon_arr = np.asarray(recon_img.convert("L"), dtype=np.float32)
        mae = float(np.abs(orig_arr - recon_arr).mean())

        side_by_side = Image.new("RGB", (nw * 2 + 10, nh), "white")
        side_by_side.paste(img_r, (0, 0))
        side_by_side.paste(recon_img, (nw + 10, 0))
        side_by_side.save(os.path.join(args.out_dir, stem + ".png"))

        results.append({
            "file_name": fname, "orig_nonzero_frac": orig_frac,
            "recon_nonzero_frac": recon_frac,
            "frac_retained_pct": 100 * recon_frac / max(orig_frac, 1e-6),
            "pixel_mae": mae,
        })
        print(f"{fname}: nonzero {orig_frac:.4f} -> {recon_frac:.4f} "
              f"({100*recon_frac/max(orig_frac,1e-6):.0f}% retained), MAE {mae:.2f}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "roundtrip_results.csv"), index=False)
    print(f"\nmean nonzero retained: {df['frac_retained_pct'].mean():.1f}%")
    worst = df.loc[df["frac_retained_pct"].idxmin()]
    print(f"worst case retained:   {worst['frac_retained_pct']:.1f}% ({worst['file_name']})")
    print(f"\nside-by-side originals|recon saved to {args.out_dir}/")
    print(
        "Look at the worst cases: if thin limbs/fingers visibly disappear or "
        "blur into blobs, consider thickening skeleton lines, adding filled "
        "joint circles, or boosting control contrast before finalizing prepare_data.py."
    )


if __name__ == "__main__":
    main()