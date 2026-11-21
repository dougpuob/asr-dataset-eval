import whisper
from .base import ASRModel

class WhisperModel(ASRModel):
    def __init__(self, config: dict):
        super().__init__(config)
        model_id = config["model"].get("id", "large-v3")
        device = config.get("endpoint", {}).get("device", "cuda")
        self.model = whisper.load_model(model_id, device=device)

    def transcribe(self, audio_path: str, language: str = None) -> str:
        lang = language or self.params.get("language")
        result = self.model.transcribe(audio_path, language=lang)
        return result["text"].strip()

    def query_info(self) -> dict:
        return {
            "status": "loaded",
            "model": self.config["model"],
            "device": self.config.get("endpoint", {}).get("device"),
        }
