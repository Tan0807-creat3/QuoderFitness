import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import QoderFitness, Layer


def bench(backend="numpy", epochs=50):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 10))
    y_dict = {
        "A": np.zeros((1000, 3)),
        "B": np.zeros((1000, 2)),
        "C": np.zeros((1000, 1)),
        "D": np.zeros((1000, 5)),
    }
    for i in range(1000):
        y_dict["A"][i, i % 3] = 1
        y_dict["B"][i, i % 2] = 1
        y_dict["C"][i, 0] = float(i % 10) / 10.0
        y_dict["D"][i, i % 5] = 1

    model = QoderFitness(backend=backend)
    model.add_shared(Layer(10, 64, "relu"))
    model.add_shared(Layer(64, 32, "relu"))
    model.add_task("A", 32, 3, "softmax", "cross_entropy", 1.0)
    model.add_task("B", 32, 2, "softmax", "cross_entropy", 1.0)
    model.add_task("C", 32, 1, "linear", "mse", 1.0)
    model.add_task("D", 32, 5, "softmax", "cross_entropy", 1.0)

    t0 = time.time()
    model.fit(X, y_dict, epochs=epochs, lr=0.03, verbose=False)
    t1 = time.time()
    print(f"{backend}: {t1 - t0:.3f} giay cho {epochs} epoch")


if __name__ == "__main__":
    bench("numpy")
    try:
        bench("torch")
    except Exception as e:
        print("torch khong chay duoc:", e)
