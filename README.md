# ASR Dataset Evaluation

獨立的 ASR 測試資料集系統。  
Air Type（或其他工具）只需把錄音丟進 `incoming/`，本系統負責管理資料集、用固定參數測試模型、儲存結果。

## 快速開始

```bash
cd asr_dataset_eval
pip install -r requirements.txt

# 1. 把音檔放進 sample/incoming/ （支援 wav / mp3 / opus / m4a）
# 2. 執行掃描，產生 metadata
python run_sample.py ingest

# 3. 手動補上正確文字後，執行測試
python run_sample.py test --config configs/sample_qwen3.json
```

## 目錄結構

```
asr_dataset_eval/
├── configs/                 # 測試設定（模型 + endpoint + 固定參數）
├── sample/
│   ├── incoming/            # Air Type 丟音檔的地方
│   ├── dataset/             # 系統管理的資料集
│   └── runs/                # 測試結果
├── asr_eval/                # 核心程式
└── run_sample.py            # 入口
```

## config.json 說明

見 `configs/sample_qwen3.json`。  
同一個 config = 同一組模型 + 同一組參數，對整個 dataset 做測試，結果會更新到對應的 runs 資料夾。
