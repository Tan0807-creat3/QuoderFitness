import json
import os
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T):
    mask = np.full((T, T), -1e9)
    for i in range(T):
        for j in range(T):
            if j <= i:
                mask[i, j] = 0.0
    return mask


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta


class Linear:
    def __init__(self, in_dim, out_dim, scale=0.02):
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)

    def __call__(self, x):
        return x @ self.W + self.b


class GELU:
    def __call__(self, x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        H = self.n_heads
        D = self.head_dim

        Q_all = self.q_proj(x)
        K_all = self.k_proj(x)
        V_all = self.v_proj(x)

        heads_out = []
        for h in range(H):
            Q = Q_all[:, :, h * D:(h + 1) * D]
            K = K_all[:, :, h * D:(h + 1) * D]
            V = V_all[:, :, h * D:(h + 1) * D]
            attn = (Q @ K.transpose(0, 2, 1)) / np.sqrt(D)
            if mask is not None:
                attn = attn + mask
            attn = softmax(attn, axis=-1)
            heads_out.append(attn @ V)

        out = np.concatenate(heads_out, axis=-1)
        return self.o_proj(out)


class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
        self.ln2 = LayerNorm(d_model)
        self.act = GELU()

    def __call__(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff2(self.act(self.ff1(self.ln2(x))))
        return x


class NanoLLM:
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=256, max_seq=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_seq, d_model) * 0.02
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.embedding[idx]
        pos = self.pos_emb[:T]
        x = tok + pos
        mask = causal_mask(T)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                vals = np.sort(logits, axis=1)[:, -top_k:]
                logits[logits < vals[:, -1:]] = -1e9
            probs = softmax(logits, axis=-1)
            next_id = np.argmax(probs, axis=-1)
            idx = np.concatenate([idx, next_id[:, None]], axis=1)
        return idx

    def save_npz(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "embedding": self.embedding,
            "pos_emb": self.pos_emb,
            "ln_f_gamma": self.ln_f.gamma,
            "ln_f_beta": self.ln_f.beta,
            "head_W": self.head.W,
            "head_b": self.head.b,
        }
        for i, block in enumerate(self.blocks):
            data[f"block{i}_linear1_W"] = block.ff1.W
            data[f"block{i}_linear1_b"] = block.ff1.b
            data[f"block{i}_linear2_W"] = block.ff2.W
            data[f"block{i}_linear2_b"] = block.ff2.b
            data[f"block{i}_self_attn_in_proj_W"] = block.attn.q_proj.W
            data[f"block{i}_self_attn_in_proj_b"] = block.attn.q_proj.b
            data[f"block{i}_self_attn_out_proj_W"] = block.attn.o_proj.W
            data[f"block{i}_self_attn_out_proj_b"] = block.attn.o_proj.b
            data[f"block{i}_norm1_weight"] = block.ln1.gamma
            data[f"block{i}_norm1_bias"] = block.ln1.beta
            data[f"block{i}_norm2_weight"] = block.ln2.gamma
            data[f"block{i}_norm2_bias"] = block.ln2.beta
        np.savez(path, **data)
        print(f"Model da luu: {path}")

    @classmethod
    def load(cls, path, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=256, max_seq=128):
        data = np.load(path)
        model = cls(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq)
        model.embedding = data["embedding"]
        model.pos_emb = data["pos_emb"]
        model.ln_f.gamma = data["ln_f_gamma"]
        model.ln_f.beta = data["ln_f_beta"]
        model.head.W = data["head_W"]
        model.head.b = data["head_b"]
        for i in range(n_layers):
            block = model.blocks[i]
            block.ff1.W = data[f"block{i}_linear1_W"]
            block.ff1.b = data[f"block{i}_linear1_b"]
            block.ff2.W = data[f"block{i}_linear2_W"]
            block.ff2.b = data[f"block{i}_linear2_b"]
            block.attn.q_proj.W = data[f"block{i}_self_attn_in_proj_W"]
            block.attn.q_proj.b = data[f"block{i}_self_attn_in_proj_b"]
            block.attn.o_proj.W = data[f"block{i}_self_attn_out_proj_W"]
            block.attn.o_proj.b = data[f"block{i}_self_attn_out_proj_b"]
            block.ln1.gamma = data[f"block{i}_norm1_weight"]
            block.ln1.beta = data[f"block{i}_norm1_bias"]
            block.ln2.gamma = data[f"block{i}_norm2_weight"]
            block.ln2.beta = data[f"block{i}_norm2_bias"]
        return model
