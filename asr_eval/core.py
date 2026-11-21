import json
from pathlib import Path
from datetime import datetime
from .utils import load_jsonl, save_jsonl, convert_to_wav, compute_cer
from .models.qwen3_asr import Qwen3ASRModel
from .models.whisper import WhisperModel

MODEL_MAP = {
    "qwen3_asr": Qwen3ASRModel,
    "whisper": WhisperModel,
}

class DatasetManager:
    def __init__(self, root: Path):
        self.root = root
        self.incoming = root / "incoming"
        self.dataset = root / "dataset"
        self.audio_dir = self.dataset / "audio"
        self.metadata_path = self.dataset / "metadata.jsonl"
        self.runs_dir = root / "runs"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.incoming.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self):
        """掃描 incoming，把新音檔納入 metadata"""
        meta = {item["id"]: item for item in load_jsonl(self.metadata_path)}
        added = 0
        for p in sorted(self.incoming.glob("*")):
            if p.suffix.lower() not in {".wav", ".mp3", ".opus", ".m4a", ".flac"}:
                continue
            sid = p.stem
            if sid in meta:
                continue
            # 複製或移動到 dataset/audio
            dest = self.audio_dir / p.name
            if not dest.exists():
                dest.write_bytes(p.read_bytes())
            meta[sid] = {
                "id": sid,
                "audio": f"audio/{p.name}",
                "text": "",          # 之後手動補正確文字
                "language": "Chinese",
                "source": "incoming",
                "added_at": datetime.now().isoformat(timespec="seconds"),
                "tags": []
            }
            added += 1
        save_jsonl(self.metadata_path, list(meta.values()))
        print(f"Ingested {added} new files. Total: {len(meta)}")
        return added

    def load_metadata(self):
        return load_jsonl(self.metadata_path)

class TestRunner:
    def __init__(self, dataset: DatasetManager):
        self.dataset = dataset

    def run(self, config_path: Path):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_id = config["run_id"]
        run_dir = self.dataset.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # 儲存完整 config
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        model_type = config["model"]["type"]
        ModelCls = MODEL_MAP.get(model_type)
        if not ModelCls:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"Loading model: {config['model']['name']} ...")
        model = ModelCls(config)
        info = model.query_info()
        (run_dir / "queried_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Model info:", info)

        meta = self.dataset.load_metadata()
        results = []
        for item in meta:
            if not item.get("text"):
                print(f"Skip {item['id']} (no reference text)")
                continue
            audio_path = self.dataset.dataset / item["audio"]
            print(f"Testing {item['id']} ...")
            wav = convert_to_wav(audio_path)
            try:
                hyp = model.transcribe(str(wav), language=item.get("language"))
            finally:
                if wav != audio_path and wav.exists():
                    wav.unlink(missing_ok=True)

            cer = compute_cer(item["text"], hyp)
            results.append({
                "id": item["id"],
                "audio": item["audio"],
                "reference": item["text"],
                "hypothesis": hyp,
                "cer": round(cer, 4),
                "model": config["model"]["name"]
            })
            print(f"  CER: {cer:.4f}")

        save_jsonl(run_dir / "results.jsonl", results)
        avg = sum(r["cer"] for r in results) / len(results) if results else 0
        print(f"Done. {len(results)} samples, avg CER: {avg:.4f}")
        print(f"Results saved to {run_dir}")
