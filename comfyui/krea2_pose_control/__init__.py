"""Standalone ComfyUI package for frozen Krea-2 Pose Control-LoRA v1."""
from .nodes import Krea2PoseCondition, Krea2PoseExtractor, Krea2PoseGenerate

NODE_CLASS_MAPPINGS = {
    "Krea2PoseExtractor": Krea2PoseExtractor,
    "Krea2PoseCondition": Krea2PoseCondition,
    "Krea2PoseGenerate": Krea2PoseGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2PoseExtractor": "Krea-2 Pose Extractor (COCO-17)",
    "Krea2PoseCondition": "Krea-2 Pose Condition (Validate)",
    "Krea2PoseGenerate": "Krea-2 Pose Generate (Turbo v1)",
}
