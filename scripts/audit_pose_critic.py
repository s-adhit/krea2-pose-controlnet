"""Run the audit-only real-image / image-gradient RTMPose checks.

This never imports train.py, opens checkpoints, or invokes an optimizer.
Run it only with locally staged official config/weights; it deliberately never
downloads a model implicitly.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

from pose_controlnet.dataset_index import DatasetIndex, ManifestRecord
from pose_controlnet.paired_preprocessing import preprocess_pair
from pose_controlnet.pose_critic import (crop_to_critic, load_official_rtmpose, pose_loss,
    sidecar_person_target, simcc_statistics)
from pose_controlnet.pose_targets import load_sidecar

SOURCES = ("coco", "humanart_painting", "humanart_real_human", "humanart_sculpture")

def tensor(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2,0,1).div(255).unsqueeze(0)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--sidecar", type=Path, required=True); p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--weights", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--per-source", type=int, default=16); p.add_argument("--device", default="cuda"); a=p.parse_args()
    if a.per_source < 16: raise ValueError("Phase-2 audit requires at least 16 samples/source")
    a.output_dir.mkdir(parents=True, exist_ok=True); _, rows=load_sidecar(a.sidecar); idx=DatasetIndex.discover(a.dataset_root)
    selected={s:[r for r in rows if r.get("source")==s and r.get("pose_reward_available") is True][:a.per_source] for s in SOURCES}
    if any(len(v)<a.per_source for v in selected.values()): raise RuntimeError(f"Insufficient eligible records: { {k:len(v) for k,v in selected.items()} }")
    critic=load_official_rtmpose(a.config,a.weights,a.device); metrics=defaultdict(lambda: defaultdict(list)); contact=[]
    for source, records in selected.items():
      for n,row in enumerate(records):
        rgb,control=idx.resolve(row["stem"]+".jpg"); pair=preprocess_pair(ManifestRecord("audit",row["stem"],row["stem"]+".jpg","audit",rgb,control))
        x=tensor(pair.rgb).to(a.device).requires_grad_(n==0); per_losses=[]
        for person in row["people"]:
          crop, target, valid=sidecar_person_target(person); valid=valid.to(a.device); target=target.to(a.device)
          logits_x,logits_y=critic(crop_to_critic(x,crop,critic.spec)); stats=simcc_statistics(logits_x,logits_y,critic.spec)
          predicted=stats["coords"][0]; err=(predicted-target).square().sum(-1).sqrt(); diag=float((person["bbox_training_xywh"][2]**2+person["bbox_training_xywh"][3]**2)**.5)
          if valid.any():
            metrics[source]["error_over_diag"].extend((err[valid]/diag).detach().cpu().tolist()); metrics[source]["confidence"].extend(stats["confidence"][0,valid].detach().cpu().tolist()); metrics[source]["entropy"].extend(stats["entropy"][0,valid].detach().cpu().tolist())
            for kind in ("expectation_huber","gaussian_cross_entropy"): metrics[source][kind].append(float(pose_loss(logits_x,logits_y,target[None],valid[None],kind=kind, spec=critic.spec).detach()))
            per_losses.append(pose_loss(logits_x,logits_y,target[None],valid[None],kind="gaussian_cross_entropy",spec=critic.spec))
        if n==0 and per_losses:
          torch.stack(per_losses).mean().backward(); metrics[source]["image_gradient_norm"].append(float(x.grad.norm().detach()))
          if not torch.isfinite(x.grad).all() or not x.grad.norm()>0: raise FloatingPointError(f"{source}: invalid image gradient")
        if len(contact)<64:
          im=pair.rgb.copy(); d=ImageDraw.Draw(im)
          for person in row["people"]: d.rectangle(tuple(person["bbox_training_xywh"][:2])+tuple(np.add(person["bbox_training_xywh"][:2],person["bbox_training_xywh"][2:])), outline="red", width=3)
          contact.append(im.resize((192,192)))
    sheet=Image.new("RGB",(8*192,8*192)); [sheet.paste(im,((i%8)*192,(i//8)*192)) for i,im in enumerate(contact)]; sheet.save(a.output_dir/"real_image_contact_sheet.jpg")
    summary={s:{k:float(np.mean(v)) if v else None for k,v in d.items()} | {"joint_count":len(d["error_over_diag"]),"pck_005":float(np.mean(np.array(d["error_over_diag"])<=.05)),"pck_010":float(np.mean(np.array(d["error_over_diag"])<=.10))} for s,d in metrics.items()}
    (a.output_dir/"real_image_audit.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
