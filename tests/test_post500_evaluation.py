import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers.modeling_outputs import BaseModelOutputWithPooling
from pose_controlnet.post500_evaluation import (CHECKPOINT_STEPS, POSE_METRIC_UNAVAILABLE_REASON, associate_people, choose_best, clip_feature_tensor, cosine_from_embeddings, export_allowlisted, pck_for_people, plot_summary, prepare_clip_scoring_inputs, unavailable_pose_result)
from pose_controlnet.reference_pose import pck_person_from_source

def person(points): return {"keypoints": points}
def joints(x=0, confidence=1): return [[x + i, 0, confidence] for i in range(17)]

class Post500EvaluationTest(unittest.TestCase):
 def test_order_pck_normalization_missing_and_low_confidence(self):
  ref=person(joints()); pred=person(joints(.5)); metric=pck_for_people([ref],[pred],.5)
  self.assertEqual(tuple(CHECKPOINT_STEPS),(0,20,40,60,80,100,200,225,350,475,500,600,700,800,900,1000,1100,1200,1300,1400,1500)); self.assertEqual(metric["evaluated_joint_count"],17); self.assertEqual(metric["pck_020"],1.0)
  missing=pck_for_people([ref],[],.5); self.assertEqual(missing["evaluated_joint_count"],17); self.assertEqual(missing["pck_005"],0.0)
  low=pck_for_people([ref],[person(joints(confidence=.1))],.5); self.assertEqual(low["evaluated_joint_count"],17); self.assertEqual(low["pck_020"],0.0)
 def test_deterministic_association_and_clip_cosine(self):
  refs=[person(joints(0)),person(joints(100))]; preds=[person(joints(101)),person(joints(1))]
  self.assertEqual(associate_people(refs,preds,.5),[(0,1),(1,0)])
  self.assertTrue(np.allclose(cosine_from_embeddings(np.array([[1,0]]),np.array([[2,0]])),[1]))
 def test_perfect_eligible_joint_prediction_is_one(self):
  source=[[float(i),0.0,2.0 if i in (5,6,7,9,11,13,15) else 0.0] for i in range(17)]
  reference=pck_person_from_source(source); metric=pck_for_people([reference],[person(reference["keypoints"])],.5)
  self.assertEqual(metric["pck_005"],1.0); self.assertEqual(metric["pck_010"],1.0); self.assertEqual(metric["pck_020"],1.0)
 def test_missing_nonrendered_joint_does_not_penalize_pck(self):
  source=[[float(i),0.0,2.0 if i in (5,6,7,9,11,13,15) else 0.0] for i in range(17)]
  reference=pck_person_from_source(source); predicted=[point.copy() for point in reference["keypoints"]]; predicted[1]=[0.0,0.0,0.0]
  self.assertEqual(pck_for_people([reference],[person(predicted)],.5)["pck_020"],1.0)
 def test_missing_rendered_joint_penalizes_coverage_and_pck(self):
  source=[[float(i),0.0,2.0 if i in (5,6,7,9,11,13,15) else 0.0] for i in range(17)]
  reference=pck_person_from_source(source); predicted=[point.copy() for point in reference["keypoints"]]; predicted[15]=[0.0,0.0,0.0]
  metric=pck_for_people([reference],[person(predicted)],.5)
  self.assertLess(metric["joint_evaluation_coverage"],1.0); self.assertLess(metric["pck_020"],1.0)
 def test_clip_tensor_and_structured_feature_returns_are_compatible(self):
  image=torch.tensor([[3.0,4.0]]); text=torch.tensor([[6.0,8.0]])
  structured=BaseModelOutputWithPooling(last_hidden_state=torch.zeros(1,1,2),pooler_output=text)
  image_embedding=clip_feature_tensor(image); text_embedding=clip_feature_tensor(structured)
  self.assertIsInstance(image_embedding,torch.Tensor); self.assertIsInstance(text_embedding,torch.Tensor); self.assertEqual(image_embedding.shape,text_embedding.shape)
  self.assertTrue(np.isfinite(cosine_from_embeddings(image_embedding.numpy(),text_embedding.numpy())).all())
  with self.assertRaises(TypeError): clip_feature_tensor(object())
 def test_clip_tokenizer_truncates_to_context_limit(self):
  class Processor:
   def __call__(self,**kwargs): self.kwargs=kwargs; return {"input_ids":torch.zeros((1,kwargs["max_length"]),dtype=torch.long)}
  processor=Processor(); caption="word "*135; context_length=77
  encoded=prepare_clip_scoring_inputs(processor,caption,Image.new("RGB",(2,2)),context_length)
  self.assertEqual(processor.kwargs["text"],[caption]); self.assertTrue(processor.kwargs["truncation"]); self.assertEqual(processor.kwargs["max_length"],context_length); self.assertEqual(encoded["input_ids"].shape[-1],context_length)
 def test_unavailable_pose_preserves_nulls_and_best_metrics(self):
  rows=[]
  for step in CHECKPOINT_STEPS: rows.append({"checkpoint_step":step,"fixed_flow":{"mean":float(500-step),"median":1,"std":0,"sample_count":1},"pose":unavailable_pose_result(),"clip":{"mean_cosine_similarity":step/500,"median_cosine_similarity":0,"std_cosine_similarity":0,"sample_count":1}})
  summary={"metadata":{"pose_metric_status":"unavailable","pose_metric_reason":POSE_METRIC_UNAVAILABLE_REASON},"checkpoints":rows}; best=choose_best(summary)
  self.assertEqual(best["lowest_fixed_flow_mean"],1500); self.assertEqual(best["highest_clip_mean_cosine_similarity"],1500)
  self.assertIsNone(best["highest_pck_005"]); self.assertIsNone(best["highest_pck_010"]); self.assertIsNone(best["highest_pck_020"]); self.assertIsNone(best["highest_detection_coverage"])
  for row in rows:
   self.assertEqual(row["pose"]["pose_metric_reason"],POSE_METRIC_UNAVAILABLE_REASON)
   self.assertIsNone(row["pose"]["pck_005"]); self.assertIsNone(row["pose"]["pck_010"]); self.assertIsNone(row["pose"]["pck_020"]); self.assertIsNone(row["pose"]["detection_coverage"])
 def test_unavailable_plots_are_skipped_and_export_excludes_them(self):
  rows=[]
  for step in CHECKPOINT_STEPS: rows.append({"checkpoint_step":step,"fixed_flow":{"mean":float(500-step),"median":1,"std":0,"sample_count":1},"pose":unavailable_pose_result(),"clip":{"mean_cosine_similarity":step/500,"median_cosine_similarity":0,"std_cosine_similarity":0,"sample_count":1}})
  summary={"metadata":{"pose_metric_status":"unavailable","pose_metric_reason":POSE_METRIC_UNAVAILABLE_REASON},"checkpoints":rows}
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp); (root/"evaluation_summary.json").write_text(json.dumps(summary)); (root/"fixed_pose").mkdir(); Image.new("RGB",(2,2)).save(root/"fixed_pose/comparison_grid.png")
   made=plot_summary(root/"evaluation_summary.json",root); self.assertEqual({x.name for x in made},{"fixed_flow_vs_step.png","clip_similarity_vs_step.png"})
   self.assertFalse((root/"pck_vs_step.png").exists()); self.assertFalse((root/"detection_coverage_vs_step.png").exists())
   destination=root/"docs"; copied=export_allowlisted(root,destination); self.assertEqual({x.name for x in copied},{"comparison_grid.png","evaluation_summary.json","fixed_flow_vs_step.png","clip_similarity_vs_step.png"})
   self.assertFalse((destination/"pck_vs_step.png").exists()); self.assertFalse((destination/"detection_coverage_vs_step.png").exists())
   Image.new("RGB",(2,2)).save(destination/"step_000500.png")
   with self.assertRaises(ValueError): export_allowlisted(root,destination)
 def test_scorer_does_not_invoke_pose_detector_or_training(self):
  scorer=(Path(__file__).resolve().parents[1]/"scripts"/"score_post500.py").read_text()
  self.assertNotIn("KeypointRCNNEstimator",scorer); self.assertNotIn("optimizer",scorer.lower())
 def test_reference_gate_does_not_run_training_or_optimizer(self):
  gate=(Path(__file__).resolve().parents[1]/"scripts"/"reference_pose_gate.py").read_text()
  self.assertNotIn("torch.optim",gate); self.assertNotIn("TrainConfig",gate); self.assertNotIn("train.py",gate)
if __name__=="__main__": unittest.main()
