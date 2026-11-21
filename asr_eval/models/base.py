from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class ASRModel(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.params = config.get("params", {})

    @abstractmethod
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        pass

    def query_info(self) -> dict:
        """可選：查詢模型資訊"""
        return {"status": "ok", "model": self.config.get("model", {})}
