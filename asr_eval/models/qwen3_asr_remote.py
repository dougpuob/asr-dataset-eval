import requests
from .base import ASRModel

class Qwen3ASRRemoteModel(ASRModel):
    """透過 HTTP 呼叫遠端 Qwen3-ASR（OpenAI-compatible endpoint）"""

    def __init__(self, config: dict):
        super().__init__(config)
        endpoint = config.get("endpoint", {})
        self.base_url = endpoint.get("url") or endpoint.get("base_url")
        if not self.base_url:
            raise ValueError("endpoint.url 或 endpoint.base_url 必須設定")
        self.base_url = self.base_url.rstrip("/")
        self.api_key = endpoint.get("api_key") or endpoint.get("auth") or "none"
        self.model_name = config["model"].get("id") or config["model"].get("name", "qwen3-asr")

    def transcribe(self, audio_path: str, language: str = None) -> str:
        url = f"{self.base_url}/v1/audio/transcriptions"
        headers = {}
        if self.api_key and self.api_key != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"

        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            data = {
                "model": self.model_name,
                "response_format": "json",
            }
            if language:
                data["language"] = language

            resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            return result.get("text", "").strip()

    def query_info(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            models = resp.json() if resp.ok else {}
        except Exception as e:
            models = {"error": str(e)}
        return {
            "status": "remote",
            "base_url": self.base_url,
            "model": self.config["model"],
            "queried": models,
        }
