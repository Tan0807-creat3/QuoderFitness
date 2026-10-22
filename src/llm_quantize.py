import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm import NanoLLM


def quantize_int8(arr):
    """Luong tuong hoa INT8 don gian: scale = max(abs) / 127."""
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = np.max(np.abs(arr)) + 1e-8
    scale = max_abs / 127.0
    q = np.round(arr / scale).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale


def quantize_model(model, save_path="models/llm_int8.npz"):
    """Luong tuong hoa toan bo trong so sang INT8 de giam bo nho."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {}

    # Embedding
    q, s = quantize_int8(model.embedding)
    data["embedding_q"] = q
    data["embedding_s"] = np.array([s], dtype=np.float32)

    # Pos emb
    q, s = quantize_int8(model.pos_emb)
    data["pos_emb_q"] = q
    data["pos_emb_s"] = np.array([s], dtype=np.float32)

    # LN
    data["ln_f_gamma"] = model.ln_f.gamma.astype(np.float32)
    data["ln_f_beta"] = model.ln_f.beta.astype(np.float32)

    # Head
    q, s = quantize_int8(model.head.W)
    data["head_W_q"] = q
    data["head_W_s"] = np.array([s], dtype=np.float32)
    data["head_b"] = model.head.b.astype(np.float32)

    # Blocks
    for i, block in enumerate(model.blocks):
        # Attn Q
        q, s = quantize_int8(block.attn.q_proj.W)
        data[f"block{i}_attn_q_W_q"] = q
        data[f"block{i}_attn_q_W_s"] = np.array([s], dtype=np.float32)
        data[f"block{i}_attn_q_b"] = block.attn.q_proj.b.astype(np.float32)

        q, s = quantize_int8(block.attn.k_proj.W)
        data[f"block{i}_attn_k_W_q"] = q
        data[f"block{i}_attn_k_W_s"] = np.array([s], dtype=np.float32)
        data[f"block{i}_attn_k_b"] = block.attn.k_proj.b.astype(np.float32)

        q, s = quantize_int8(block.attn.v_proj.W)
        data[f"block{i}_attn_v_W_q"] = q
        data[f"block{i}_attn_v_W_s"] = np.array([s], dtype=np.float32)
        data[f"block{i}_attn_v_b"] = block.attn.v_proj.b.astype(np.float32)

        q, s = quantize_int8(block.attn.o_proj.W)
        data[f"block{i}_attn_o_W_q"] = q
        data[f"block{i}_attn_o_W_s"] = np.array([s], dtype=np.float32)
        data[f"block{i}_attn_o_b"] = block.attn.o_proj.b.astype(np.float32)

        # LN1
        data[f"block{i}_ln1_gamma"] = block.ln1.gamma.astype(np.float32)
        data[f"block{i}_ln1_beta"] = block.ln1.beta.astype(np.float32)

        # FF1
        q, s = quantize_int8(block.ff1.W)
        data[f"block{i}_ff1_W_q"] = q
        data[f"block{i}_ff1_W_s"] = np.array([s], dtype=np.float32)
        data[f"block{i}_ff1_b"] = block.ff1.b.astype(np.float32)

        # FF2
        q, s = quantize_int8(block.ff2.W)
        data[f"block{i}_ff2_W_q"] = q
        data[f"block{i}_ff2_W_s"] = np.array([s], dtype=np.float32)
        data[f"block{i}_ff2_b"] = block.ff2.b.astype(np.float32)

        # LN2
        data[f"block{i}_ln2_gamma"] = block.ln2.gamma.astype(np.float32)
        data[f"block{i}_ln2_beta"] = block.ln2.beta.astype(np.float32)

    np.savez(save_path, **data)
    print(f"Model INT8 da luu: {save_path}")
    print("Giam bo nho khoang 4x so voi float32.")


if __name__ == "__main__":
    from src.llm import NanoLLM

    import json
    with open("configs/model_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    llm_cfg = cfg["llm"]

    model = NanoLLM(
        vocab_size=llm_cfg["vocab_size"],
        d_model=llm_cfg["d_model"],
        n_heads=llm_cfg["n_heads"],
        n_layers=llm_cfg["n_layers"],
        d_ff=llm_cfg["d_ff"],
        max_seq=llm_cfg["max_seq"],
    )
    model.load("models/llm.npz", vocab_size=llm_cfg["vocab_size"],
               d_model=llm_cfg["d_model"], n_heads=llm_cfg["n_heads"],
               n_layers=llm_cfg["n_layers"], d_ff=llm_cfg["d_ff"],
               max_seq=llm_cfg["max_seq"])

    quantize_model(model, "models/llm_int8.npz")
