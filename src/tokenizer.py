import json
import os
import re
from collections import Counter


class CharTokenizer:
    """Tokenizer ki tu + tu, nhe, khong can thu vien ngoai. Chay duoc tren CPU may thuong."""

    def __init__(self, vocab=None, special_tokens=None):
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            " uine": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        if vocab is None:
            self.vocab = dict(self.special_tokens)
        else:
            self.vocab = dict(vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.word_freq = Counter()

    def _preprocess(self, text):
        # Tach tu va ki tu dac biet, giu lai dau cach
        text = text.replace(" ", " <sp> ")
        return text

    def fit(self, texts, min_freq=1, max_vocab=5000):
        counter = Counter()
        for t in texts:
            counter.update(list(t))
        # Them cac tu xuat hien nhieu
        word_counter = Counter()
        for t in texts:
            words = re.findall(r"\S+", t)
            word_counter.update(words)
        for ch, freq in counter.items():
            if freq >= min_freq and ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        for word, freq in word_counter.most_common(max_vocab):
            if word not in self.vocab and freq >= min_freq:
                self.vocab[word] = len(self.vocab)
        self.word_freq = word_counter
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.vocab["<bos>"])
        # Tu dau tien: thao tac chuoi ky tu
        i = 0
        while i < len(text):
            matched = False
            # Tham do toi da 5 ki tu de match tu
            for L in range(min(5, len(text) - i), 0, -1):
                sub = text[i:i + L]
                if sub in self.vocab:
                    ids.append(self.vocab[sub])
                    i += L
                    matched = True
                    break
            if not matched:
                ids.append(self.vocab.get(text[i], self.vocab[" uine"]))
                i += 1
        if add_eos:
            ids.append(self.vocab["<eos>"])
        return ids

    def decode(self, ids, skip_special=False):
        chars = []
        for i in ids:
            if skip_special and i in (
                self.vocab["<pad>"],
                self.vocab["<bos>"],
                self.vocab["<eos>"],
                self.vocab[" uine"],
            ):
                continue
            chars.append(self.id_to_token.get(i, " uine"))
        # <sp> -> khoang cach
        text = "".join(chars)
        text = text.replace("<sp>", " ")
        return text

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "special_tokens": self.special_tokens,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["vocab"], special_tokens=data.get("special_tokens"))
