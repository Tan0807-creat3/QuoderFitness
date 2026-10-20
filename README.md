# QoderFitness - AI da tac vu (tong quat)

Kien truc: Mat chia chung (shared layers) + cac task head rieng cho moi tac vu.
Viet boi Python + NumPy/PyTorch, chay tot tren CPU may thuong, huat luyen nhanh tren GPU VPS.

## Cau truc chuan (manifest v2.1.0)

```
qoderfitness/
├── configs/model_config.json      # Manifest chinh (nhanh nhu Hugging Face model card)
├── src/
│   ├── __init__.py
│   ├── model.py                   # QoderFitness core (multi-task, tu manifest)
│   ├── data.py                    # Data generation & split
│   ├── train.py                   # Training loop (doc manifest)
│   ├── infer.py                   # Inference (doc manifest)
│   └── utils.py                   # Scaler, metrics, logger, performance config
├── test/
│   ├── test_env.py                # Moi truong test AI
│   └── benchmark.py               # So sanh toc do
├── models/
│   ├── qoderfitness_best.json     # Mo hinh toan bo (trong so + cau hinh)
│   ├── qoderfitness_encoder.json  # Chi luu shared layers (transfer learning)
│   └── x_scaler.npz               # Thang so hoa dau vao
├── logs/                          # Training logs
├── scripts/
│   ├── train_vps.ps1              # Chay treo may tren VPS GPU
│   └── infer_client.bat           # Chay tren may van phong CPU
├── requirements.txt               # Thu vien can thiet
├── README.md                      # Tai lieu
├── ROADMAP.md                     # Huong phat trien
└── .venv/                         # Moi truong ao
```

## Tac vu ho tro

- **A**: Phan loai 3 lop (softmax + cross_entropy)
- **B**: Phan loai 2 lop (softmax + cross_entropy)
- **C**: Hoi quy 1 gia tri thuc (linear + mse)
- **D**: Phan loai 5 lop (softmax + cross_entropy)

## Cai dat

```powershell
cd qoderfitness
pip install -r requirements.txt
```

## Huat luyen (VPS GPU hoac CPU)

```powershell
# Chay trong venv
.venv\Scripts\Activate.ps1
pip install torch  # hoac torch CPU neu khong co GPU
# Sua configs/model_config.json: "device": "cuda" neu co GPU, "cpu" neu khong
.venv\Scripts\python.exe src/train.py
```

Hoan toan duoc dieu chinh qua file `configs/model_config.json` (manifest) khong can sua code.

## Test AI (may van phong, khong can GPU)

```powershell
.venv\Scripts\python.exe test/test_env.py
```

Lenh ho tro:
- Nhap 10 so thuc -> du doan 4 tac vu
- `demo` -> chay 5 mau ngau nhien
- `file duong_dan` -> doc nhieu hang tu file txt
- `quit` -> thoat

## Chuyen backend

Sua `configs/model_config.json` -> `"training" -> "device"` thanh `"cuda"` neu co GPU NVIDIA, hoac `"cpu"` (mac dinh) cho CPU co MKLDNN.

## Toi uu hieu suat

- Multi-threading CPU (so luong thread tu dong)
- MKLDNN bat mac dinh tren CPU Intel
- Gradient accumulation (configurable)
- Learning rate warmup + cosine annealing
- Early stopping voi min_delta
- Mixed precision san sang neu co CUDA
- Model save/load chua thiet bi (GPU-trained -> CPU-inference)
