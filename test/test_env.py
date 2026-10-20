import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.infer import load_model_for_inference, predict_single


def main():
    print("===== Moi truong test QoderFitness (multi-task) =====")
    print("Nhap 10 so thuc cach boi khoang trang, hoac lenh:")
    print("  demo    - chay 5 mau ngau nhien")
    print("  file duong_dan  - doc nhieu hang tu file txt (moi hang 10 so)")
    print("  quit    - thoat")
    print("Vi du: 0.5 -1.2 0.3 1.0 0.0 -0.4 2.1 0.7 -0.9 1.3")
    print()

    model = load_model_for_inference()

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break

        if line.lower() == "demo":
            rng = np.random.default_rng(123)
            for i in range(5):
                x = rng.standard_normal((1, 10))
                res = predict_single(model, x=x[0])
                print(f"[Demo {i+1}]")
                for k, v in res.items():
                    if k == "C":
                        print(f"  {k} (hoi quy): {v['value']:.4f}")
                    else:
                        print(f"  {k} (lop {v['label']}): xac suat {np.round(v['probs'], 3)}")
                print()
            continue

        if line.lower().startswith("file "):
            path = line[5:].strip()
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for ln, row in enumerate(f, 1):
                        row = row.strip()
                        if not row:
                            continue
                        try:
                            vals = [float(v) for v in row.split()]
                            if len(vals) != 10:
                                print(f"Hang {ln}: phai co 10 so, bo qua.")
                                continue
                            res = predict_single(model, x=vals)
                            print(f"[Hang {ln}]")
                            for k, v in res.items():
                                if k == "C":
                                    print(f"  {k} (hoi quy): {v['value']:.4f}")
                                else:
                                    print(f"  {k} (lop {v['label']}): xac suat {np.round(v['probs'], 3)}")
                            print()
                        except ValueError:
                            print(f"Hang {ln}: sai dinh dang, bo qua.")
                continue
            except FileNotFoundError:
                print("Khong tim thay file.")
                continue

        try:
            vals = [float(v) for v in line.split()]
            if len(vals) != 10:
                print("Phai nhap dung 10 so.")
                continue
        except ValueError:
            print("Nhap khong hop le.")
            continue

        res = predict_single(model, x=vals)
        print("Ket qua:")
        for k, v in res.items():
            if k == "C":
                print(f"  {k} (hoi quy): {v['value']:.4f}")
            else:
                print(f"  {k} (lop {v['label']}): xac suat {np.round(v['probs'], 3)}")
        print()


if __name__ == "__main__":
    main()
