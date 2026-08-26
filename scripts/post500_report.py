"""Plot, print, or allowlist-export the post-500 evaluation summary."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.post500_evaluation import export_allowlisted, plot_summary
def main():
 p=argparse.ArgumentParser(); p.add_argument("action",choices=("plots","report","export")); p.add_argument("--output-dir",default="/lambda/nfs/adhit/krea2-pose/evaluation/pose-learning-500"); p.add_argument("--destination",default="docs/evaluation/pose-learning-500"); a=p.parse_args(); root=Path(a.output_dir)
 if a.action=="plots": print(*plot_summary(root/"evaluation_summary.json",root),sep="\n")
 elif a.action=="export": print(*export_allowlisted(root,a.destination),sep="\n")
 else:
  summary=json.loads((root/"evaluation_summary.json").read_text()); rows=summary["checkpoints"]
  if summary.get("metadata",{}).get("pose_metric_status")=="unavailable": print("PCK unavailable — authoritative reference pose missing")
  print("Flow MSE lower is better; CLIP higher is better"); print("Step | Flow MSE | PCK@.05 | PCK@.10 | PCK@.20 | CLIP")
  for x in rows: print(f"{x['checkpoint_step']:>4} | {x['fixed_flow']['mean']:.6f} | {x['pose']['pck_005'] if x['pose']['pck_005'] is not None else 'N/A'} | {x['pose']['pck_010'] if x['pose']['pck_010'] is not None else 'N/A'} | {x['pose']['pck_020'] if x['pose']['pck_020'] is not None else 'N/A'} | {x['clip']['mean_cosine_similarity']:.6f}")
if __name__=="__main__": main()
