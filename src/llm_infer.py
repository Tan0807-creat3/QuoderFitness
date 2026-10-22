import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tokenizer import CharTokenizer
from src.llm import NanoLLM


def load_model_for_inference(
    config_path="configs/model_config.json",
    model_path="models/llm.npz",
    tokenizer_path="models/tokenizer.json",
):
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    llm_cfg = cfg["llm"]
    tokenizer = CharTokenizer.load(tokenizer_path)

    model = NanoLLM(
        vocab_size=llm_cfg["vocab_size"],
        d_model=llm_cfg["d_model"],
        n_heads=llm_cfg["n_heads"],
        n_layers=llm_cfg["n_layers"],
        d_ff=llm_cfg["d_ff"],
        max_seq=llm_cfg["max_seq"],
    )
    model.load(model_path, vocab_size=llm_cfg["vocab_size"], d_model=llm_cfg["d_model"],
               n_heads=llm_cfg["n_heads"], n_layers=llm_cfg["n_layers"],
               d_ff=llm_cfg["d_ff"], max_seq=llm_cfg["max_seq"])
    print(f"Model da tai: {model_path}")
    print(f"Tokenizer da tai: {tokenizer_path} (vocab_size={tokenizer.vocab_size})")
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt="Chao",
    max_new_tokens=50,
    temperature=0.8,
    top_k=None,
):
    ids = tokenizer.encode(prompt)
    idx = np.array([ids], dtype=np.int64)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    generated = out[0].tolist()
    return tokenizer.decode(generated, skip_special=True)


if __name__ == "__main__":
    model, tokenizer = load_model_for_inference()

    tests = ["Chao", "Toi yeu", "Python la", "Tri tue", "QoderFitness"]
    for prompt in tests:
        text = generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8)
        print(f"Input: '{prompt}'")
        print(f"Output: '{text}'")
        print("-" * 40)

    print("\nNhap prompt de sinh van ban (hoac 'quit' de thoat):")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        text = generate_text(model, tokenizer, line, max_new_tokens=50, temperature=0.8)
        print(f"> {text}")
        print()
