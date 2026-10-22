import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tokenizer import CharTokenizer
from src.unified_model import QoderFitnessUnified


def load_model_for_inference(
    config_path="configs/model_config.json",
    model_path="models/unified_best.npz",
    tokenizer_path="models/tokenizer.json",
):
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    llm_cfg = cfg["llm"]
    tokenizer = CharTokenizer.load(tokenizer_path)

    model = QoderFitnessUnified(
        vocab_size=llm_cfg["vocab_size"],
        d_model=llm_cfg["d_model"],
        n_heads=llm_cfg["n_heads"],
        n_layers=llm_cfg["n_layers"],
        d_ff=llm_cfg["d_ff"],
        max_seq=llm_cfg["max_seq"],
    )
    model.load(model_path,
               vocab_size=llm_cfg["vocab_size"],
               d_model=llm_cfg["d_model"],
               n_heads=llm_cfg["n_heads"],
               n_layers=llm_cfg["n_layers"],
               d_ff=llm_cfg["d_ff"],
               max_seq=llm_cfg["max_seq"])
    print(f"Unified model đã tải: {model_path}")
    print(f"Tokenizer đã tải: {tokenizer_path} (vocab_size={tokenizer.vocab_size})")
    return model, tokenizer


def predict_tasks(model, tokenizer, text):
    """
    Nhập văn bản, trả về kết quả tất cả tác vụ.
    """
    ids = tokenizer.encode(text)
    idx = np.array([ids], dtype=np.int64)
    outs = model.predict(idx)

    def softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    result = {
        "task_A_3_lop": int(np.argmax(outs["A"][0])),
        "task_B_2_lop": int(np.argmax(outs["B"][0])),
        "task_C_hoi_quy": float(outs["C"][0][0]),
        "task_D_5_lop": int(np.argmax(outs["D"][0])),
        "probs_A": outs["A"][0].tolist(),
        "probs_B": outs["B"][0].tolist(),
        "probs_D": outs["D"][0].tolist(),
    }
    return result


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=None):
    ids = tokenizer.encode(prompt)
    idx = np.array([ids], dtype=np.int64)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    generated = out[0].tolist()
    return tokenizer.decode(generated, skip_special=True)


def chat_loop():
    print("=" * 50)
    print("QoderFitness Unified AI - Chat trên CPU")
    print("Nhập 'quit' để thoát, 'clear' để xóa màn hình")
    print("=" * 50)

    import json
    with open("configs/model_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    llm_cfg = cfg["llm"]

    model = QoderFitnessUnified(
        vocab_size=llm_cfg["vocab_size"],
        d_model=llm_cfg["d_model"],
        n_heads=llm_cfg["n_heads"],
        n_layers=llm_cfg["n_layers"],
        d_ff=llm_cfg["d_ff"],
        max_seq=llm_cfg["max_seq"],
    )
    model.load("models/unified_best.npz",
               vocab_size=llm_cfg["vocab_size"],
               d_model=llm_cfg["d_model"],
               n_heads=llm_cfg["n_heads"],
               n_layers=llm_cfg["n_layers"],
               d_ff=llm_cfg["d_ff"],
               max_seq=llm_cfg["max_seq"])
    tokenizer = CharTokenizer.load("models/tokenizer.json")

    while True:
        try:
            line = input("\nBạn: ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            print("Tạm biệt!")
            break
        if line.lower() == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        # Phân tích tác vụ
        res = predict_tasks(model, tokenizer, line)
        print(f"  [Phân loại] A: {res['task_A_3_lop']}, B: {res['task_B_2_lop']}, D: {res['task_D_5_lop']}")
        print(f"  [Hồi quy] C: {res['task_C_hoi_quy']:.4f}")

        # Sinh văn bản
        resp = generate_text(model, tokenizer, line, max_new_tokens=40, temperature=0.8)
        print(f"  [AI]: {resp}")


if __name__ == "__main__":
    chat_loop()
