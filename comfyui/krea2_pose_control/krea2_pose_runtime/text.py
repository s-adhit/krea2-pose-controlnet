"""Frozen Qwen3-VL text-conditioning contract for Krea-2."""
from __future__ import annotations
import torch


class PoseTextConditioner(torch.nn.Module):
    PREFIX = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    PREFIX_IDX, SELECT_LAYERS = 34, (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, max_length: int = 512):
        super().__init__()
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", torch_dtype=dtype).to(device).eval().requires_grad_(False)
        self.tokenizer, self.device, self.max_length = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct"), device, max_length

    @torch.no_grad()
    def forward(self, prompts: list[str]):
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("a non-empty prompt list is required")
        contexts, masks = [], []
        suffix = self.tokenizer([self.SUFFIX], return_tensors="pt").to(self.device)
        for prompt in prompts:
            inputs = self.tokenizer([self.PREFIX + prompt], truncation=True, padding="longest",
                                    max_length=self.max_length + self.PREFIX_IDX, return_tensors="pt",
                                    padding_side="right").to(self.device)
            ids = torch.cat([inputs["input_ids"], suffix["input_ids"]], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix["attention_mask"].bool()], dim=1)
            output = self.qwen(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            contexts.append(torch.stack([output.hidden_states[i] for i in self.SELECT_LAYERS], dim=2)[0, self.PREFIX_IDX:])
            masks.append(mask[0, self.PREFIX_IDX:])
        length = max(context.shape[0] for context in contexts)
        return (torch.stack([torch.nn.functional.pad(context, (0, 0, 0, 0, 0, length - context.shape[0])) for context in contexts]),
                torch.stack([torch.nn.functional.pad(mask, (0, length - mask.shape[0])) for mask in masks]))
