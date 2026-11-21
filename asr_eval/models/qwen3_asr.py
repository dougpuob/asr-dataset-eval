import torch
from .base import ASRModel

class Qwen3ASRModel(ASRModel):
    def __init__(self, config: dict):
        super().__init__(config)
        from qwen_asr import Qwen3ASRModel as _Qwen
        model_id = config["model"].get("id", "Qwen/Qwen3-ASR-1.7B")
        device = config.get("endpoint", {}).get("device", "cuda:0")
        dtype = self.params.get("dtype", "bfloat16")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        self.model = _Qwen.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map=device,
        )

    def transcribe(self, audio_path: str, language: str = None) -> str:
        lang = language or self.params.get("language")
        results = self.model.transcribe(audio=audio_path, language=lang)
        return results[0].text if results else ""

    def query_info(self) -> dict:
        return {
            "status": "loaded",
            "model": self.config["model"],
            "device": self.config.get("endpoint", {}).get("device"),
        }
