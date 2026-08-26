"""text_encoder.py — Qwen3-VL text conditioner for Krea-2's context tower."""
import torch


class PoseTextConditioner(torch.nn.Module):
    PREFIX = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    PREFIX_IDX = 34
    SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", max_length=512,
                 device="cuda", dtype=torch.bfloat16):
        super().__init__()
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        self.qwen = (
            Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
            .to(device).eval().requires_grad_(False)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_length = max_length
        self.device = device

    @torch.no_grad()
    def forward(self, prompts: list[str]):
        if not prompts:
            raise ValueError("PoseTextConditioner requires at least one prompt")

        # Encode each caption independently.  The former mixed-length batch path
        # inserted right-padding between a short prompt and the appended suffix;
        # that made a cache entry's valid-token layout differ from the online
        # conditioning contract.  Per-caption encoding makes each suffix adjacent
        # to its prompt, then this method restores only trailing batch padding.
        contexts: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        suffix = self.tokenizer([self.SUFFIX], return_tensors="pt").to(self.device)
        for prompt in prompts:
            inputs = self.tokenizer(
                [self.PREFIX + prompt], truncation=True, padding="longest",
                max_length=self.max_length + self.PREFIX_IDX,
                return_tensors="pt", padding_side="right",
            ).to(self.device)
            input_ids = torch.cat([inputs["input_ids"], suffix["input_ids"]], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix["attention_mask"].bool()], dim=1)
            out = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
            hidden = torch.stack([out.hidden_states[i] for i in self.SELECT_LAYERS], dim=2)
            contexts.append(hidden[0, self.PREFIX_IDX:])
            masks.append(mask[0, self.PREFIX_IDX:])

        length = max(context.shape[0] for context in contexts)
        return (
            torch.stack([torch.nn.functional.pad(context, (0, 0, 0, 0, 0, length - context.shape[0])) for context in contexts]),
            torch.stack([torch.nn.functional.pad(mask, (0, length - mask.shape[0])) for mask in masks]),
        )
