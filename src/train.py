import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.model import QoderFitness
from src.data import generate_multitask_dataset, train_test_split
from src.utils import (
    save_scaler, accuracy_onehot, rmse, compute_score, set_seed, configure_performance, setup_logger
)


def train_model(config_path="configs/model_config.json"):
    print("[DEBUG] train_model started", flush=True)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    set_seed(cfg["training"]["seed"])
    configure_performance(cfg.get("training", {}).get("device", "cpu"), cfg)
    logger = setup_logger("qoderfitness", cfg.get("paths", {}).get("log_dir", "logs"))

    data_cfg = cfg["data"]
    print("[DEBUG] Generating dataset...", flush=True)
    X, y = generate_multitask_dataset(
        n_samples=data_cfg["n_samples"], seed=cfg["training"]["seed"]
    )
    print("[DEBUG] Dataset generated; splitting...", flush=True)
    X_train, y_train, X_test, y_test = train_test_split(
        X, y, test_ratio=data_cfg["test_ratio"], seed=cfg["training"]["seed"]
    )

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std

    print("[DEBUG] Data prepared; creating model...", flush=True)
    backend = cfg.get("training", {}).get("device", "cpu")
    model = QoderFitness(config_path=config_path, backend=backend)
    model.summary()
    print("[DEBUG] Model created; entering training loop", flush=True)

    print("[TRAIN] Starting a single training run...", flush=True)
    model.fit(X_train_n, y_train, verbose=True)

    print("[EVAL] Evaluating trained model...", flush=True)
    outs = model.predict(X_test_n)

    aA = accuracy_onehot(outs["A"], y_test["A"])
    aB = accuracy_onehot(outs["B"], y_test["B"])
    rC = rmse(outs["C"], y_test["C"])
    aD = accuracy_onehot(outs["D"], y_test["D"])

    best = compute_score(aA, aB, rC, aD)
    msg = (
        f"Final metrics | A:{aA:.4f} B:{aB:.4f} "
        f"C(rmse):{rC:.4f} D:{aD:.4f} | score:{best:.6f}"
    )
    print(msg, flush=True)
    logger.info(msg)

    model.save(cfg.get("paths", {}).get(
        "model_save", "models/qoderfitness_best.json"
    ))

    save_scaler(X_mean, X_std, cfg.get("paths", {}).get("scaler_save", "models/x_scaler.npz"))
    model.save_encoder(cfg.get("paths", {}).get("encoder_save", "models/qoderfitness_encoder.json"))
    print(f"Da luu qoderfitness_best.json | best score {best:.4f}", flush=True)
    print("Da luu encoder vao qoderfitness_encoder.json", flush=True)
    logger.info(f"Training complete. Best score: {best:.4f}")
    return model, best


if __name__ == "__main__":
    train_model()
