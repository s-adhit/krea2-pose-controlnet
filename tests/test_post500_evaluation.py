import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from pose_controlnet.post500_evaluation import (CHECKPOINT_STEPS, associate_people, choose_best, cosine_from_embeddings, export_allowlisted, pck_for_people, plot_summary)

def person(points): return {"keypoints": points}
def joints(x=0, confidence=1): return [[x + i, 0, confidence] for i in range(17)]

class Post500EvaluationTest(unittest.TestCase):
 def test_order_pck_normalization_missing_and_low_confidence(self):
  ref=person(joints()); pred=person(joints(.5)); metric=pck_for_people([ref],[pred],.5)
  self.assertEqual(tuple(CHECKPOINT_STEPS),(0,20,40,60,80,100,200,225,350,475,500)); self.assertEqual(metric["evaluated_joint_count"],17); self.assertEqual(metric["pck_020"],1.0)
  missing=pck_for_people([ref],[],.5); self.assertEqual(missing["evaluated_joint_count"],0); self.assertEqual(missing["pck_005"],None)
  low=pck_for_people([ref],[person(joints(confidence=.1))],.5); self.assertEqual(low["evaluated_joint_count"],0)
 def test_deterministic_association_and_clip_cosine(self):
  refs=[person(joints(0)),person(joints(100))]; preds=[person(joints(101)),person(joints(1))]
  self.assertEqual(associate_people(refs,preds,.5),[(0,1),(1,0)])
  self.assertTrue(np.allclose(cosine_from_embeddings(np.array([[1,0]]),np.array([[2,0]])),[1]))
 def test_best_plots_and_allowlist(self):
  rows=[]
  for step in CHECKPOINT_STEPS: rows.append({"checkpoint_step":step,"fixed_flow":{"mean":float(500-step),"median":1,"std":0,"sample_count":1},"pose":{"pck_005":step/500,"pck_010":step/500,"pck_020":step/500,"detection_coverage":step/500},"clip":{"mean_cosine_similarity":step/500,"median_cosine_similarity":0,"std_cosine_similarity":0,"sample_count":1}})
  summary={"checkpoints":rows}; self.assertEqual(choose_best(summary)["lowest_fixed_flow_mean"],500)
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp); (root/"evaluation_summary.json").write_text(json.dumps(summary)); (root/"fixed_pose").mkdir(); Image.new("RGB",(2,2)).save(root/"fixed_pose/comparison_grid.png")
   self.assertEqual(len(plot_summary(root/"evaluation_summary.json",root)),4); destination=root/"docs"; copied=export_allowlisted(root,destination); self.assertTrue(all("step_" not in x.name for x in copied))
   Image.new("RGB",(2,2)).save(destination/"step_000500.png")
   with self.assertRaises(ValueError): export_allowlisted(root,destination)
if __name__=="__main__": unittest.main()
