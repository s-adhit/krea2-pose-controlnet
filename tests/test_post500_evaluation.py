import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers.modeling_outputs import BaseModelOutputWithPooling
from pose_controlnet.post500_evaluation import (CHECKPOINT_STEPS, associate_people, choose_best, clip_feature_tensor, cosine_from_embeddings, export_allowlisted, pck_for_people, plot_summary, prepare_clip_scoring_inputs)

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
