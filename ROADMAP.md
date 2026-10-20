# Roadmap phat trien QoderFitness (AI da tac vu)

## Da hoan thanh (v2.1.0)

- Mang nho viet tu boi NumPy/PyTorch (forward, backward, train, save/load JSON manifest).
- Kien truc da tac vu: mat chia chung + cac task head rieng.
- Ho tro 4 tac vu: phan loai 3 lop, phan loai 2 lop, hoi quy, phan loai 5 lop.
- Manifest config (configs/model_config.json) tuong tu Hugging Face model card.
- Moi truong test command line (test/test_env.py).
- Benchmark toc do (test/benchmark.py).
- Toi uu CPU: MKLDNN, multi-thread, gradient clipping, cosine LR schedule.
- Model checkpointing (best score), encoder chuan bi transfer learning.
- Ho tro train tren GPU VPS, infer tren CPU may van phong.

## Buoc ke tiep (ke hoach)

1. **Fine-tuning / Transfer learning**: su dung `models/qoderfitness_encoder.json` lam backbone co dinh, them dau ra cho nhieu tac vu moi.
2. **Du lieu thuc**: thay du lieu ngau nhien bang du lieu thuc (CSV/JSON) cho bai tao thuc su.
3. **Xuat mo hinh**: ONNX Runtime / PyTorch `.pt` de trien khai tren cac thiet bi khac.
4. **API / Ứng dụng**: wrap thanh REST API hoac ung dung desktop de nguoi dung nhap du lieu va nhan ket qua.
5. **Hyperparameter search**: them Optuna hoac grid search tu manifest config.

## Lenh chay

```powershell
cd qoderfitness
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.venv\Scripts\python.exe src/train.py
.venv\Scripts\python.exe test/test_env.py
```
