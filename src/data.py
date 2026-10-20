import numpy as np


def generate_multitask_dataset(n_samples=50000, seed=2026):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 10))

    wA = rng.standard_normal((10, 3))
    yA = np.argmax(X @ wA, axis=1)

    wB = rng.standard_normal((10, 2))
    yB = np.argmax(X @ wB, axis=1)

    wC = rng.standard_normal((10, 1))
    yC = (X @ wC).flatten() + 0.1 * rng.standard_normal(n_samples)

    wD = rng.standard_normal((10, 5))
    yD = np.argmax(X @ wD, axis=1)

    def onehot(idx, n):
        o = np.zeros((idx.shape[0], n))
        o[np.arange(idx.shape[0]), idx] = 1
        return o

    y_dict = {
        "A": onehot(yA, 3),
        "B": onehot(yB, 2),
        "C": yC.reshape(-1, 1),
        "D": onehot(yD, 5),
    }
    return X, y_dict


def train_test_split(X, y_dict, test_ratio=0.1, seed=2026):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_test = int(X.shape[0] * test_ratio)
    train_idx = idx[n_test:]
    test_idx = idx[:n_test]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = {k: v[train_idx] for k, v in y_dict.items()}
    y_test = {k: v[test_idx] for k, v in y_dict.items()}
    return X_train, y_train, X_test, y_test
