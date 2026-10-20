import os
import json
import logging
import numpy as np
from datetime import datetime


def setup_logger(name="qoderfitness", log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        sh = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scaler(mean, std, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=np.array(mean, dtype=np.float32), std=np.array(std, dtype=np.float32))


def load_scaler(path):
    data = np.load(path)
    return data["mean"], data["std"]


def accuracy_onehot(pred, true):
    p = np.argmax(pred, axis=1)
    t = np.argmax(true, axis=1)
    return float(np.mean(p == t))


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_score(accA, accB, rmseC, accD):
    return (1.0 - accA) + (1.0 - accB) + rmseC * 0.1 + (1.0 - accD)


def set_seed(seed):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_performance(device="cpu", cfg=None):
    import torch
    perf = (cfg or {}).get("performance", {})
    if device == "cpu":
        threads = perf.get("num_threads", 8)
        interop = perf.get("interop_threads", 4)
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(interop)
        if perf.get("mkldnn", True) and hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
    if perf.get("deterministic", True):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
