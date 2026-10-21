import os
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


def activate(Z, act):
    if act == "relu":
        return np.maximum(0.0, Z)
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500.0, 500.0)))
    if act == "tanh":
        return np.tanh(Z)
    if act == "softmax":
        e = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    if act == "linear":
        return Z
    return Z


def activate_derivative(Z, act):
    if act == "relu":
        return (Z > 0).astype(float)
    if act == "sigmoid":
        s = activate(Z, "sigmoid")
        return s * (1.0 - s)
    if act == "tanh":
        return 1.0 - np.tanh(Z) ** 2
    if act == "softmax":
        return np.ones_like(Z)
    if act == "linear":
        return np.ones_like(Z)
    return np.ones_like(Z)


def cross_entropy(y_pred, y_true):
    eps = 1e-9
    return -np.mean(np.sum(y_true * np.log(np.clip(y_pred, eps, 1.0)), axis=1))


def d_cross_entropy(y_pred, y_true):
    return (y_pred - y_true) / y_pred.shape[0]


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def d_mse(y_pred, y_true):
    return 2.0 * (y_pred - y_true) / y_pred.shape[0]


class Layer:
    def __init__(self, input_size, output_size, activation="relu", W=None, b=None):
        if W is None:
            self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        else:
            self.W = np.array(W, dtype=np.float32)
        self.b = np.zeros((1, output_size), dtype=np.float32) if b is None else np.array(b, dtype=np.float32)
        self.activation = activation
        self.cache = {}

    def forward(self, X):
        self.cache["X"] = X
        Z = X @ self.W + self.b
        self.cache["Z"] = Z
        return activate(Z, self.activation), Z


class TaskHead:
    def __init__(self, input_size, output_size, activation, loss, weight=1.0, W=None, b=None):
        self.layer = Layer(input_size, output_size, activation, W, b)
        self.activation = activation
        self.loss = loss
        self.weight = weight
        self.cache = {}

    def forward(self, X):
        A, Z = self.layer.forward(X)
        self.cache["X"] = X
        self.cache["Z"] = Z
        return A

    def backward(self, dA, lr):
        X = self.cache["X"]
        Z = self.cache["Z"]
        dZ = dA * activate_derivative(Z, self.layer.activation)
        dW = X.T @ dZ / X.shape[0]
        db = np.sum(dZ, axis=0, keepdims=True) / X.shape[0]
        dX = dZ @ self.layer.W.T
        self.layer.W -= lr * dW
        self.layer.b -= lr * db
        return dX


class TorchBackend:
    def __init__(self, cfg):
        import torch
        self.torch = torch
        self.cfg = cfg
        perf = cfg.get("performance", {})
        device_name = cfg.get("training", {}).get("device", "cpu")

        # Uu tien: TPU (XLA) > CUDA > CPU
        if device_name == "tpu" and XLA_AVAILABLE:
            try:
                self.device = xm.xla_device()
                self.name = "torch (tpu/xla)"
                self._is_tpu = True
            except Exception:
                self.device = torch.device("cpu")
                self.name = "torch (cpu, fallback from tpu)"
                self._is_tpu = False
        elif device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.name = "torch (cuda)"
            self._is_tpu = False
        elif device_name == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
            self.name = "torch (xpu)"
            self._is_tpu = False
        else:
            self.device = torch.device("cpu")
            self.name = "torch (cpu, mkldnn)"
            self._is_tpu = False
            try:
                torch.set_num_threads(perf.get("num_threads", 8))
                torch.set_num_interop_threads(perf.get("interop_threads", 4))
                if perf.get("mkldnn", True) and hasattr(torch.backends, "mkldnn"):
                    torch.backends.mkldnn.enabled = True
                if perf.get("deterministic", True):
                    torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

        self.shared = None
        self.heads = None
        self.task_configs = None
        self.task_order = None
        self.built = False

    def _build(self, shared_sizes, task_configs, task_order=None):
        t = self.torch
        shared_modules = []
        prev = shared_sizes[0]
        for i, sz in enumerate(shared_sizes[1:]):
            shared_modules.append((f"lin{i}", t.nn.Linear(prev, sz)))
            shared_modules.append((f"act{i}", t.nn.ReLU()))
            prev = sz
        self.shared = t.nn.Sequential(*[m for _, m in shared_modules])

        self.heads = t.nn.ModuleDict()
        for name, cfg in task_configs.items():
            head_modules = [t.nn.Linear(prev, cfg["out"])]
            act = cfg.get("activation", "linear")
            if act == "relu":
                head_modules.append(t.nn.ReLU())
            elif act == "sigmoid":
                head_modules.append(t.nn.Sigmoid())
            elif act == "tanh":
                head_modules.append(t.nn.Tanh())
            self.heads[name] = t.nn.Sequential(*head_modules)

        self.shared.to(self.device)
        self.heads.to(self.device)
        self.task_configs = task_configs
        self.task_order = task_order if task_order is not None else list(task_configs.keys())
        self.built = True

    def to_numpy(self, tensor):
        if self._is_tpu:
            import torch_xla.core.xla_model as xm
            return tensor.cpu().detach().numpy()
        return tensor.detach().cpu().numpy()

    def predict(self, X):
        t = self.torch
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim == 0:
            X = X.reshape(1, 1)
        Xt = t.tensor(X, dtype=t.float32, device=self.device)
        with t.no_grad():
            A = self.shared(Xt)
            outs = {}
            for name in self.task_configs:
                out = self.heads[name](A)
                if self.task_configs[name].get("activation") == "softmax":
                    out = t.nn.functional.softmax(out, dim=1)
                outs[name] = self.to_numpy(out)
        return outs

    def fit(self, X, y_dict, epochs, lr, clip, verbose,
            optimizer="adam", weight_decay=0.0, lr_schedule="cosine",
            batch_size=256, warmup_epochs=10, accumulation_steps=1):
        t = self.torch
        if self._is_tpu:
            import torch_xla.core.xla_model as xm

        Xt = t.tensor(X, dtype=t.float32, device=self.device)
        y_ts = {k: t.tensor(v, dtype=t.float32, device=self.device) for k, v in y_dict.items()}

        params = list(self.shared.parameters()) + [p for h in self.heads.values() for p in h.parameters()]
        if optimizer.lower() == "adam":
            opt = t.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            opt = t.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)

        if lr_schedule == "cosine":
            sched = t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
        elif lr_schedule == "step":
            sched = t.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)
        else:
            sched = None

        ce = t.nn.CrossEntropyLoss(label_smoothing=0.1)
        mse_fn = t.nn.MSELoss()

        labels = {}
        for name, cfg in self.task_configs.items():
            y = y_dict[name]
            if cfg["loss"] == "cross_entropy":
                labels[name] = t.tensor(np.argmax(y, axis=1), dtype=t.long, device=self.device)
            else:
                labels[name] = y_ts[name]

        scaler = None

        task_loss_ema = {name: 1.0 for name in self.task_order}
        ema_alpha = 0.98

        n = X.shape[0]
        for epoch in range(epochs):
            perm = t.randperm(n)
            Xt_shuf = Xt[perm]
            total_loss = 0.0
            nb = 0

            for i in range(0, n, batch_size):
                xb = Xt_shuf[i:i + batch_size]
                yb_labels = {k: labels[k][perm[i:i + batch_size]] for k in labels}
                yb_raw = {k: y_ts[k][perm[i:i + batch_size]] for k in y_ts}

                opt.zero_grad()
                A = self.shared(xb)
                total_loss = 0.0
                for name, cfg in self.task_configs.items():
                    out = self.heads[name](A)
                    if cfg["loss"] == "cross_entropy":
                        loss = ce(out, yb_labels[name])
                    else:
                        loss = mse_fn(out, yb_raw[name])
                    task_loss_ema[name] = ema_alpha * task_loss_ema[name] + (1 - ema_alpha) * loss.item()
                    norm_loss = loss / (task_loss_ema[name] + 1e-8)
                    total_loss = total_loss + self._task_weights[name] * norm_loss

                total_loss.backward()
                if self._is_tpu:
                    xm.optimizer_step(opt)
                else:
                    t.nn.utils.clip_grad_norm_(params, clip)
                    opt.step()

                total_loss += total_loss.item()
                nb += 1

            if sched is not None:
                sched.step()

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs} - total_loss: {total_loss / max(nb, 1):.4f} [{self.name}] lr={opt.param_groups[0]['lr']:.6f}")

    def get_weights(self):
        t = self.torch
        data = {"shared": [], "tasks": {}}
        for i, m in enumerate(self.shared):
            if isinstance(m, t.nn.Linear):
                data["shared"].append({
                    "input_size": int(m.in_features),
                    "output_size": int(m.out_features),
                    "activation": "relu",
                    "W": self.to_numpy(m.weight.T).tolist(),
                    "b": self.to_numpy(m.bias).tolist(),
                })
        for name, head in self.heads.items():
            cfg = self.task_configs[name]
            lin = head[0] if len(head) > 0 else None
            if lin is None:
                continue
            data["tasks"][name] = {
                "name": name,
                "input_size": int(lin.in_features),
                "output_size": int(lin.out_features),
                "activation": cfg.get("activation", "linear"),
                "loss": cfg["loss"],
                "weight": self._task_weights[name],
                "W": self.to_numpy(lin.weight.T).tolist(),
                "b": self.to_numpy(lin.bias).tolist(),
            }
        return data


class QoderFitness:
    def __init__(self, config_path="configs/model_config.json", backend=None):
        import json as _json
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = _json.load(f)

        if backend is None:
            import torch
            if XLA_AVAILABLE:
                backend = "tpu"
            elif torch.cuda.is_available():
                backend = "cuda"
            else:
                backend = "cpu"

        if backend in ("tpu", "cuda", "xpu", "gpu"):
            self.backend_name = "torch"
            self.backend = TorchBackend(self.cfg) if TORCH_AVAILABLE else None
        elif backend in ("cpu", "torch"):
            self.backend_name = "torch"
            self.backend = TorchBackend(self.cfg) if TORCH_AVAILABLE else None
        elif backend == "numpy":
            self.backend_name = "numpy"
            self.backend = None
        else:
            raise ValueError("backend phai la 'numpy', 'torch', 'cpu', 'cuda', 'tpu' hoac 'xpu'")

        self.arch = self.cfg["architecture"]
        self.train_cfg = self.cfg["training"]
        self.data_cfg = self.cfg["data"]
        self.perf_cfg = self.cfg.get("performance", {})
        self.paths_cfg = self.cfg.get("paths", {})

        self.shared_layers = []
        self.task_heads = {}
        self.task_order = []
        self.loss_weights = {}
        self.loss_fns = {
            "mse": (mse, d_mse),
            "cross_entropy": (cross_entropy, d_cross_entropy),
        }

        self._build_from_config()

    def _build_from_config(self):
        input_dim = self.arch["input_dim"]
        prev = input_dim
        for sz in self.arch["shared_layers"]:
            self.add_shared(Layer(prev, sz, self.arch["shared_activation"]))
            prev = sz
        for name, cfg in self.arch["task_heads"].items():
            self.add_task(
                name, prev, cfg["output_dim"],
                activation=cfg["activation"],
                loss=cfg["loss"],
                weight=cfg.get("weight", 1.0),
            )

    def add_shared(self, layer):
        self.shared_layers.append(layer)

    def add_task(self, name, input_size, output_size, activation="softmax", loss="cross_entropy", weight=1.0):
        self.task_heads[name] = TaskHead(input_size, output_size, activation, loss, weight)
        self.task_order.append(name)
        self.loss_weights[name] = weight

    def _build_torch(self):
        if self.backend is None or self.backend.built:
            return
        shared_sizes = [self.arch["input_dim"]] + [l.W.shape[1] for l in self.shared_layers]
        task_configs = {}
        for name in self.task_order:
            h = self.task_heads[name]
            task_configs[name] = {
                "out": int(h.layer.W.shape[1]),
                "activation": h.activation,
                "loss": h.loss,
            }
        self.backend._task_weights = self.loss_weights
        self.backend._build(shared_sizes, task_configs, task_order=self.task_order)
        t = self.backend.torch
        with t.no_grad():
            idx = 0
            for m in self.backend.shared:
                if isinstance(m, t.nn.Linear):
                    layer = self.shared_layers[idx]
                    m.weight.copy_(t.tensor(layer.W.T, dtype=t.float32, device=self.backend.device))
                    m.bias.copy_(t.tensor(layer.b[0], dtype=t.float32, device=self.backend.device))
                    idx += 1
            for name in self.task_order:
                head = self.backend.heads[name]
                lin = head[0]
                h = self.task_heads[name]
                lin.weight.copy_(t.tensor(h.layer.W.T, dtype=t.float32, device=self.backend.device))
                lin.bias.copy_(t.tensor(h.layer.b[0], dtype=t.float32, device=self.backend.device))

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim == 0:
            X = X.reshape(1, 1)

        if self.backend is not None:
            try:
                self._build_torch()
                return self.backend.predict(X)
            except Exception:
                self.backend = None

        A, _ = self._forward_shared(X)
        outs = {}
        for name in self.task_order:
            outs[name] = self.task_heads[name].forward(A)
        return outs

    def _forward_shared(self, X):
        A = np.array(X, dtype=np.float32)
        if A.ndim == 1:
            A = A.reshape(1, -1)
        caches = []
        for layer in self.shared_layers:
            A_new, Z = layer.forward(A)
            caches.append((layer, A, Z))
            A = A_new
        return A, caches

    def fit(self, X, y_dict, verbose=True):
        X = np.array(X, dtype=np.float32)
        if self.backend is not None:
            self._build_torch()
            self.backend.fit(
                X, y_dict,
                epochs=self.train_cfg["epochs"],
                lr=self.train_cfg["learning_rate"],
                clip=self.train_cfg["gradient_clip"],
                verbose=verbose,
                optimizer=self.train_cfg["optimizer"],
                weight_decay=self.train_cfg["weight_decay"],
                lr_schedule=self.train_cfg["lr_schedule"],
                batch_size=self.train_cfg["batch_size"],
                warmup_epochs=self.train_cfg.get("warmup_epochs", 10),
                accumulation_steps=self.train_cfg.get("gradient_accumulation_steps", 1),
            )
            self._sync_torch_to_numpy()
            return

        for epoch in range(self.train_cfg["epochs"]):
            A, scaches = self._forward_shared(X)
            head_outs = {}
            for name in self.task_order:
                head_outs[name] = self.task_heads[name].forward(A)

            total_loss = 0.0
            grads = {}
            for name in self.task_order:
                y_true = np.array(y_dict[name], dtype=np.float32)
                y_pred = head_outs[name]
                loss_fn, dloss_fn = self.loss_fns[self.task_heads[name].loss]
                l = loss_fn(y_pred, y_true)
                total_loss += self.loss_weights[name] * l
                grads[name] = dloss_fn(y_pred, y_true) * self.loss_weights[name]

            dA = np.zeros_like(A)
            for name in self.task_order:
                d_head = self.task_heads[name].backward(grads[name], self.train_cfg["learning_rate"])
                dA = dA + d_head

            for i in range(len(self.shared_layers) - 1, -1, -1):
                layer, Xin, Zin = scaches[i]
                dZ = dA * activate_derivative(Zin, layer.activation)
                dW = Xin.T @ dZ / Xin.shape[0]
                db = np.sum(dZ, axis=0, keepdims=True) / Xin.shape[0]
                dA = dZ @ layer.W.T
                np.clip(dW, -self.train_cfg["gradient_clip"], self.train_cfg["gradient_clip"], out=dW)
                np.clip(db, -self.train_cfg["gradient_clip"], self.train_cfg["gradient_clip"], out=db)
                layer.W -= self.train_cfg["learning_rate"] * dW
                layer.b -= self.train_cfg["learning_rate"] * db

            if verbose and (epoch % 10 == 0 or epoch == self.train_cfg["epochs"] - 1):
                print(f"Epoch {epoch + 1}/{self.train_cfg['epochs']} - total_loss: {total_loss:.4f}")

    def _sync_torch_to_numpy(self):
        t = self.backend.torch
        idx = 0
        with t.no_grad():
            for m in self.backend.shared:
                if isinstance(m, t.nn.Linear):
                    layer = self.shared_layers[idx]
                    layer.W = self.backend.to_numpy(m.weight.T).astype(np.float32)
                    layer.b = self.backend.to_numpy(m.bias).reshape(1, -1).astype(np.float32)
                    idx += 1
            for name in self.task_order:
                head = self.backend.heads[name]
                lin = head[0]
                h = self.task_heads[name]
                h.layer.W = self.backend.to_numpy(lin.weight.T).astype(np.float32)
                h.layer.b = self.backend.to_numpy(lin.bias).reshape(1, -1).astype(np.float32)

    def summary(self):
        print("QoderFitness multi-task model (manifest v2.1.0)")
        print("=" * 50)
        print(f"Backend: {self.backend.name if self.backend else 'numpy'}")
        print(f"Input dim: {self.arch['input_dim']}")
        print("Shared layers:")
        for i, layer in enumerate(self.shared_layers, 1):
            print(f"  {i}: {layer.W.shape[0]} -> {layer.W.shape[1]} | {layer.activation}")
        print("Task heads:")
        for name in self.task_order:
            h = self.task_heads[name]
            print(f"  {name}: {h.layer.W.shape[0]} -> {h.layer.W.shape[1]} | {h.activation} | {h.loss} | w={self.loss_weights[name]}")
        print("=" * 50)

    def save(self, path=None):
        if path is None:
            path = self.paths_cfg.get("model_save", "models/qoderfitness_best.json")
        data = {
            "backend": self.backend_name,
            "config": self.cfg,
            "shared": [
                {
                    "input_size": int(l.W.shape[0]),
                    "output_size": int(l.W.shape[1]),
                    "activation": l.activation,
                    "W": l.W.tolist(),
                    "b": l.b.tolist(),
                }
                for l in self.shared_layers
            ],
            "tasks": [],
        }
        for name in self.task_order:
            h = self.task_heads[name]
            data["tasks"].append({
                "name": name,
                "input_size": int(h.layer.W.shape[0]),
                "output_size": int(h.layer.W.shape[1]),
                "activation": h.activation,
                "loss": h.loss,
                "weight": self.loss_weights[name],
                "W": h.layer.W.tolist(),
                "b": h.layer.b.tolist(),
            })
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_encoder(self, path=None):
        if path is None:
            path = self.paths_cfg.get("encoder_save", "models/qoderfitness_encoder.json")
        data = {
            "backend": self.backend_name,
            "type": "encoder",
            "config": self.cfg,
            "shared": [
                {
                    "input_size": int(l.W.shape[0]),
                    "output_size": int(l.W.shape[1]),
                    "activation": l.activation,
                    "W": l.W.tolist(),
                    "b": l.b.tolist(),
                }
                for l in self.shared_layers
            ],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path, config_path="configs/model_config.json", backend=None):
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        model = cls(config_path=config_path, backend=backend)
        model.shared_layers = []
        model.task_heads = {}
        model.task_order = []
        model.loss_weights = {}
        for sd in data["shared"]:
            layer = Layer(sd["input_size"], sd["output_size"], sd["activation"], W=sd["W"], b=sd["b"])
            model.add_shared(layer)
        for td in data["tasks"]:
            model.add_task(
                td["name"], td["input_size"], td["output_size"],
                td["activation"], td["loss"], td.get("weight", 1.0),
            )
            h = model.task_heads[td["name"]]
            h.layer.W = np.array(td["W"], dtype=np.float32)
            h.layer.b = np.array(td["b"], dtype=np.float32)
        return model
