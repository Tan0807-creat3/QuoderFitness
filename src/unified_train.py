import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from src.tokenizer import CharTokenizer
from src.unified_model import QoderFitnessUnified


def default_texts():
    return [
        "Chao mung den voi QoderFitness AI.",
        "Day la mo hinh nguon ngau nho chay tren CPU may thuong.",
        "Toi yeu lap trinh va tri tue nhan tao.",
        "Xin chao, ban khoe khong?",
        "QoderFitness ho tro nhieu tac vu: phan loai, hoi quy, sinh van ban.",
        "Transformer la mo hinh nguon ngon ngu hieu qua.",
        "Python la ngon ngu lap trinh pho bien trong AI.",
        "Machine learning giup may tinh hoc du lieu tu du lieu.",
        "Deep learning su dung mang nho nhieu lop de xu ly du lieu phuc tap.",
        "Toi muon hoc ve AI va lap trinh.",
        "Chat bot co the tu tra loi cau hoi don gian.",
        "Ngon ngu lap trinh cho rat nhieu ung dung thuc te.",
        "Tri tue nhan tao se thay doi cuoc song con nguoi.",
        "Hoc may la mot trong cac linh dao cua tri tue nhan tao.",
        "Toi dang hoc viet code Python.",
        "QoderFitness khong phai model the duc, la AI da tac vu.",
        "Model nhe, chay duoc tren may nhan vien thuong.",
        "Co the chay tren CPU, khong can GPU.",
        "QoderFitness ho tro nhieu tac vu cung luc.",
        "Toi thich hoc lap trinh va AI.",
    ]


def load_text(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def train_unified(
    config_path="configs/model_config.json",
    text_path=None,
    epochs=200,
    lr=0.005,
    batch_size=16,
    seq_len=64,
    save_path="models/unified_best.npz",
    tokenizer_path="models/tokenizer.json",
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    llm_cfg = cfg["llm"]
    vocab_size = llm_cfg["vocab_size"]
    d_model = llm_cfg["d_model"]
    n_heads = llm_cfg["n_heads"]
    n_layers = llm_cfg["n_layers"]
    d_ff = llm_cfg["d_ff"]
    max_seq = llm_cfg["max_seq"]

    # Chuẩn bị văn bản
    text = load_text(text_path)
    if text is None:
        text = "\n".join(default_texts())

    # Huấn luyện tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit([text], min_freq=1, max_vocab=llm_cfg["vocab_size"])
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer đã lưu: {tokenizer_path} (vocab_size={tokenizer.vocab_size})")

    # Mã hóa
    ids = tokenizer.encode(text)
    ids = [min(i, vocab_size - 1) for i in ids]
    ids = np.array(ids, dtype=np.int64)
    total_tokens = len(ids)
    print(f"Tổng số token: {total_tokens}, seq_len: {seq_len}")

    # Tạo batch (next-token prediction)
    data = []
    step = max(1, seq_len // 2)
    for i in range(0, total_tokens - seq_len, step):
        chunk = ids[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        data.append((chunk[:-1], chunk[1:]))
    if len(data) == 0:
        print("Không đủ dữ liệu để huấn luyện.")
        return

    X = np.array([d[0] for d in data], dtype=np.int64)
    Y = np.array([d[1] for d in data], dtype=np.int64)
    print(f"Số mẫu huấn luyện: {len(X)}")

    # Khởi tạo model unified
    model = QoderFitnessUnified(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq)

    # Optimizer Adam đơn giản (NumPy)
    params = {
        "embedding": model.embedding,
        "pos_emb": model.pos_emb,
        "ln_f_gamma": model.ln_f.gamma,
        "ln_f_beta": model.ln_f.beta,
        "head_class_3_W": model.head_class_3.W,
        "head_class_3_b": model.head_class_3.b,
        "head_class_2_W": model.head_class_2.W,
        "head_class_2_b": model.head_class_2.b,
        "head_reg_W": model.head_reg.W,
        "head_reg_b": model.head_reg.b,
        "head_class_5_W": model.head_class_5.W,
        "head_class_5_b": model.head_class_5.b,
        "head_lm_W": model.head_lm.W,
        "head_lm_b": model.head_lm.b,
    }
    for i, block in enumerate(model.blocks):
        params[f"block{i}_ff1_W"] = block.ff1.W
        params[f"block{i}_ff1_b"] = block.ff1.b
        params[f"block{i}_ff2_W"] = block.ff2.W
        params[f"block{i}_ff2_b"] = block.ff2.b
        params[f"block{i}_attn_q_W"] = block.attn.q_proj.W
        params[f"block{i}_attn_q_b"] = block.attn.q_proj.b
        params[f"block{i}_attn_k_W"] = block.attn.k_proj.W
        params[f"block{i}_attn_k_b"] = block.attn.k_proj.b
        params[f"block{i}_attn_v_W"] = block.attn.v_proj.W
        params[f"block{i}_attn_v_b"] = block.attn.v_proj.b
        params[f"block{i}_attn_o_W"] = block.attn.o_proj.W
        params[f"block{i}_attn_o_b"] = block.attn.o_proj.b
        params[f"block{i}_ln1_gamma"] = block.ln1.gamma
        params[f"block{i}_ln1_beta"] = block.ln1.beta
        params[f"block{i}_ln2_gamma"] = block.ln2.gamma
        params[f"block{i}_ln2_beta"] = block.ln2.beta

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_loss = float("inf")

    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        epoch_loss = 0.0
        nb = 0

        for bi in range(0, len(X), batch_size):
            idxb = perm[bi:bi + batch_size]
            xb = X[idxb]
            yb = Y[idxb]

            # Forward
            logits_dict = model.predict(xb)
            logits = logits_dict["E"]  # (B, T, vocab)
            logits_flat = logits.reshape(-1, vocab_size)
            y_flat = yb.reshape(-1)

            # Cross entropy
            logits_shift = logits_flat - logits_flat.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits_shift)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            loss = -np.log(probs[np.arange(seq_len * len(xb)), y_flat] + 1e-9).mean()
            epoch_loss += loss

            # Backward: dlogits
            dlogits = probs.copy()
            dlogits[np.arange(seq_len * len(xb)), y_flat] -= 1.0
            dlogits /= (seq_len * len(xb))

            # Gradient cho head_lm (chỉ vị trí cuối của mỗi sample)
            dlogits_last = dlogits.reshape(len(xb), seq_len, vocab_size)[:, -1, :]  # (B, vocab)
            last_emb = model.embedding[xb[:, -1]]  # (B, d_model)
            d_head_lm_W = last_emb.T @ dlogits_last  # (d_model, vocab)
            d_head_lm_b = dlogits_last.sum(axis=0)   # (vocab,)

            # Gradient cho embedding (xấp xỉ)
            d_embedding = np.zeros_like(model.embedding)
            for b_idx in range(len(xb)):
                for t in range(seq_len):
                    tok_id = xb[b_idx, t]
                    d_embedding[tok_id] += d_head_lm_W[:, y_flat[b_idx * seq_len + t]] * 0.01

            # Adam update cho các tham số chính
            for name in params:
                if name == "embedding":
                    g = d_embedding
                elif name == "head_lm_W":
                    g = d_head_lm_W
                elif name == "head_lm_b":
                    g = d_head_lm_b
                else:
                    g = np.zeros_like(params[name])

                m[name] = beta1 * m[name] + (1 - beta1) * g
                v[name] = beta2 * v[name] + (1 - beta2) * (g ** 2)
                m_hat = m[name] / (1 - beta1 ** (epoch + 1))
                v_hat = v[name] / (1 - beta2 ** (epoch + 1))
                params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            model.embedding = params["embedding"]
            model.head_lm.W = params["head_lm_W"]
            model.head_lm.b = params["head_lm_b"]

            nb += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/max(1,nb):.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model.save(save_path)

    model.save(save_path)
    print(f"Huấn luyện Unified model đã hoàn thành. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_unified()
