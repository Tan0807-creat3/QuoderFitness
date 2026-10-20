import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.model import QoderFitness
from src.utils import load_scaler


def load_model_for_inference(config_path="configs/model_config.json",
                             model_path="models/qoderfitness_best.json",
                             backend="cpu"):
    model = QoderFitness.load(model_path, config_path=config_path, backend=backend)
    return model


def predict_single(model, scaler_path="models/x_scaler.npz", x=None):
    if x is None:
        x = [0.5, -1.2, 0.3, 1.0, 0.0, -0.4, 2.1, 0.7, -0.9, 1.3]
    X_mean, X_std = load_scaler(scaler_path)
    x = np.array(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    xn = (x - X_mean) / X_std
    outs = model.predict(xn)

    result = {}
    result["A"] = {"label": int(np.argmax(outs["A"][0])), "probs": outs["A"][0].tolist()}
    result["B"] = {"label": int(np.argmax(outs["B"][0])), "probs": outs["B"][0].tolist()}
    result["C"] = {"value": float(outs["C"][0][0])}
    result["D"] = {"label": int(np.argmax(outs["D"][0])), "probs": outs["D"][0].tolist()}
    return result


def predict_batch(model, scaler_path="models/x_scaler.npz", X=None):
    if X is None:
        X = np.array([[0.5, -1.2, 0.3, 1.0, 0.0, -0.4, 2.1, 0.7, -0.9, 1.3]])
    X_mean, X_std = load_scaler(scaler_path)
    X = np.array(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    Xn = (X - X_mean) / X_std
    outs = model.predict(Xn)

    results = []
    for i in range(len(X)):
        results.append({
            "A": {"label": int(np.argmax(outs["A"][i])), "probs": outs["A"][i].tolist()},
            "B": {"label": int(np.argmax(outs["B"][i])), "probs": outs["B"][i].tolist()},
            "C": {"value": float(outs["C"][i][0])},
            "D": {"label": int(np.argmax(outs["D"][i])), "probs": outs["D"][i].tolist()},
        })
    return results


if __name__ == "__main__":
    model = load_model_for_inference()
    res = predict_single(model)
    print("Ket qua du doan (mau mac dinh):")
    for k, v in res.items():
        if k == "C":
            print(f"  {k} (hoi quy): {v['value']:.4f}")
        else:
            print(f"  {k} (lop {v['label']}): xac suat {np.round(v['probs'], 3)}")
