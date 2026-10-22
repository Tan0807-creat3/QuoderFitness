import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.tokenizer import CharTokenizer


class NanoLLMTorch(nn.Module):
    """Nano Transformer decoder-only, chay tot tren CPU may thuong."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=256, max_seq=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(max_seq, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, mask=None):
        B, T = idx.shape
        tok = self.embedding(idx)
        pos = self.pos_emb[:T]
        x = tok + pos
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def save_npz(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "embedding": self.embedding.weight.detach().cpu().numpy(),
            "pos_emb": self.pos_emb.detach().cpu().numpy(),
            "ln_f_gamma": self.ln_f.weight.detach().cpu().numpy(),
            "ln_f_beta": self.ln_f.bias.detach().cpu().numpy(),
            "head_W": self.head.weight.detach().cpu().numpy(),
            "head_b": self.head.bias.detach().cpu().numpy(),
        }
        for i, block in enumerate(self.blocks):
            data[f"block{i}_linear1_W"] = block.linear1.weight.detach().cpu().numpy()
            data[f"block{i}_linear1_b"] = block.linear1.bias.detach().cpu().numpy()
            data[f"block{i}_linear2_W"] = block.linear2.weight.detach().cpu().numpy()
            data[f"block{i}_linear2_b"] = block.linear2.bias.detach().cpu().numpy()
            data[f"block{i}_self_attn_in_proj_W"] = block.self_attn.in_proj_weight.detach().cpu().numpy()
            data[f"block{i}_self_attn_in_proj_b"] = block.self_attn.in_proj_bias.detach().cpu().numpy()
            data[f"block{i}_self_attn_out_proj_W"] = block.self_attn.out_proj.weight.detach().cpu().numpy()
            data[f"block{i}_self_attn_out_proj_b"] = block.self_attn.out_proj.bias.detach().cpu().numpy()
            data[f"block{i}_norm1_weight"] = block.norm1.weight.detach().cpu().numpy()
            data[f"block{i}_norm1_bias"] = block.norm1.bias.detach().cpu().numpy()
            data[f"block{i}_norm2_weight"] = block.norm2.weight.detach().cpu().numpy()
            data[f"block{i}_norm2_bias"] = block.norm2.bias.detach().cpu().numpy()
        np.savez(path, **data)
        print(f"Model da luu: {path}")


def load_text(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def default_texts():
    return [
        "Chao mung den voi QoderFitness AI.",
        "Day la mo hinh nguon ngau nho chay tren CPU may thuong.",
        "Toi yeu lap trinh va tri tue nhan tao.",
        "Xin chao, ban khoe khong?",
        "QoderFitness ho tro nhieu tac vu: phan loai, hoi quy, sinh van ban.",
        "Transformer la mo hinh nguon ngon ngu hieu qua.",
        "Day la mot cau du lieu huon luyen mau de kiem tra mo hinh.",
        "Python la ngon ngu lap trinh pho bien trong AI.",
        "Machine learning giup may tinh hoc du lieu tu du lieu.",
        "Deep learning su dung mang nho nhieu lop de xu ly du lieu phuc tap.",
        "Toi muon hoc ve AI va lap trinh.",
        "Chat bot co the tu tra loi cau hoi don gian.",
        "Ngon ngu lap trinh cho rat nhieu ung dung thuc te.",
        "Tri tue nhan tao se thay doi cuoc song con nguoi.",
        "Hoc may la mot trong cac linh dao cua tri tue nhan tao.",
        "Toi dang hoc viet code Python.",
        "Mat tong la mot cai dat cua mang nho.",
        "Backpropagation giup cap nhat trong so mang nho.",
        "Gradient descent la thuat toan toi uu cot ban.",
        "ReLu la ham kich hoat pho bien trong mang nho.",
        "Softmax dang o cuoi mang de phan loai.",
        "Cross entropy la ham mat thuong dung trong phan loai.",
        "Adam la optimizer tui duong toc toc.",
        "Dropout giup chong qua khop trong mang nho.",
        "Batch normalization chu bien du lieu dau vao.",
        "Transfer learning su dung mo hinh da duoc huon luyen.",
        "Fine-tuning chinh chinh chinh mo hinh cho nhiem vu cu the.",
        "Data augmentation tang bo du lieu huan luyen.",
        "Overlap la khi mo hinh hoc dat qua tren du lieu train.",
        "Underlap la khi mo hinh chua hoc duoc du lieu.",
        "Validation kiem tra mo hinh tren du lieu chua thay.",
        "Test kiem tra hieu suat cuoi cung cua mo hinh.",
        "Learning rate thiet toc toc cua gradient descent.",
        "Epoch la mot vong lap qua bo du lieu toan bo.",
        "Batch la mot lop du lieu duoc xu ly cung luc.",
        "Early stopping dung muon khi mo hinh khong tien bo.",
        "Checkpoint luu trang thai mo hinh giua chuong trinh.",
        "Tensor la du lieu dau vao cua mang nho.",
        "Gradient la dao ham cua loss theo trong so.",
        "Backprop lan truyen loi nguoc lai qua mang nho.",
        "Forward lan truyen tien tu dau den cuoi mang nho.",
        "Activation function gui ra gia tri cua mot nho.",
        "Hidden layer la lop giua trong mang nho.",
        "Input layer la lop nhan du lieu dau vao.",
        "Output layer la lop tra ket qua cuoi cung.",
        "Weight la trong so cua mo ket giua cac nho.",
        "Bias la cong them trong ham kich hoat.",
        "Loss function do kho cua mo hinh.",
        "Accuracy la ti le du doan chinh xac.",
        "Precision la ti le du doan duong trong cac du doan duong.",
        "Recall la ti le tim duoc cac truong hop duong.",
        "F1 score la trung binh cong tong hop precision va recall.",
        "Confusion matrix bang thong ke ket qua phan loai.",
        "ROC curve duong cong cong cua phan loai nhi nhat.",
        "AUC dien tich duoi cong cong ROC.",
        "K-means thuat toan gom cum khong co giam sat.",
        "Pham vi thuat toan gom cum co giam sat.",
        "Decision tree cay quyet don gian.",
        "Random forest nhieu cay quyet ket hop lai.",
        "Gradient boosting tat dan cac moc cai tien.",
        "XGBoost thuat toan boosting hieu qua.",
        "LightGBM thuat toan boosting nhanh hon.",
        "CatBoost xu ly du lieu phan loai tot.",
        "Support vector machine du doan duong thang phan cach.",
        "K-nearest neighbor tim cac hang xom gan nhat.",
        "Naive Bayes du doan dua tren xac suat co dieu kien.",
        "Logistic regression du doan xac suat phan loai nhi nhat.",
        "Linear regression du doan gia tri thuc.",
        "Polynomial regression du dao duong cong.",
        "Time series du bao gia tri thoi gian.",
        "Forecasting du bao tuong lai tu du lieu liet ke.",
        "Anomaly detection phat hien lai ngoai.",
        "Clustering gom cum cac doi tuong tuong tu.",
        "Dimensionality reduction giam chieu rong du lieu.",
        "PCA chuan hoa thanh phan chinh.",
        "t-SNE hien thi du lieu cao chieu tren 2D.",
        "UMAP giam chieu rong du lieu nhanh hon t-SNE.",
        "Autoencoder ma hoa va giai ma du lieu.",
        "Variational autoencoder tao ra du lieu moi.",
        "Generative adversarial network tran dau hai mang.",
        "Generator tao ra du lieu moi.",
        "Dam generator phan biet du lieu thuc va tao.",
        "Convolutional neural network xu ly anh.",
        "Convolution layer phep chong chap.",
        "Pooling layer giam kich thuoc du lieu.",
        "Max window chon gia tri lon nhat.",
        "Average window trung binh kich thuoc.",
        "Flatten lap day mang chieu cao thap.",
        "Dropout ngau nen tat cac mot so nho.",
        "Batch norm chuan hoa moi batch du lieu.",
        "Layer norm chuan hoa moi layer du lieu.",
        "Group norm gom cum nhom layer chuan hoa.",
        "Instance norm chuan hoa moi hinh anh.",
        "Weight norm chuan hoa trong so.",
        "Spectral norm chuan hoa gia tri don cua ma tran.",
        "L1 regularization tong tuyet doi trong so.",
        "L2 regularization tong binh phuong trong so.",
        "Elastic net ket hop L1 va L2.",
        "Early stopping dung muon khi mo hinh khong tien bo.",
        "Learning rate schedule doi toc toc trong qua trinh huon luyen.",
        "Warm-up tang toc toc bat dau chua.",
        "Cosine annealing giam toc toc theo cosin.",
        "Step decay giam toc toc theo buoc buoc.",
        "Exponential decay giam toc toc mu dan.",
        "Cyclic learning rate dao dong toc toc.",
        "One cycle policy mot vong toc toc.",
        "Lookahead optimizer cap nhat nhieu buoc.",
        "RMSprop duong cong bien doi toc toc.",
        "Adagrad duong cong tong luong gradient.",
        "Adadelta duong cong bien doi RMSprop.",
        "AdaMax duong cong bien doi Adam.",
        "Nadam duong cong ket hop Adam va Nesterov.",
        "AdaBound duong cong gioi han toc toc.",
        "Yogi duong cong bien doi Adam.",
        "DiffGrad duong cong bien doi Adam.",
        "Lamb duong cong toi uu batch lon.",
        "LambW duong cong Lambda voi weight decay.",
        "Sophia duong cong toi uu tieu chuan.",
        "SophiaG duong cong giai thuat.",
        "Marin duong cong toi uu.",
        "Muon duong cong toi uu nhieu tang.",
        "MuonSGD duong cong SGD duoc toi uu.",
        "Lion duong cong toi uu moi.",
        "Adafactor duong cong giam bo nho.",
        "AdaHessian duong cong toi uu cap hai.",
        "KFAC duong cong tien bien ma tran.",
        "Shampoo duong cong tien bien ma tran.",
        "Diagonal duong cong duong cheo.",
        "Block diagonal duong cong cac block duong cheo.",
        "Kron duong cong tich Kron.",
        "Hadamard duong cong tich phan tu.",
        "Outer product tich ngoai cua hai vector.",
        "Inner product tich noi cua hai vector.",
        "Dot product tich vo huong cua hai vector.",
        "Cross product tich vo huong cua hai vector.",
        "Matrix multiplication nhan ma tran.",
        "Matrix transposition chuyen vi ma tran.",
        "Matrix inverse ma tran nghich dao.",
        "Matrix determinant dinh thuc ma tran.",
        "Matrix trace duong cheo chinh ma tran.",
        "Matrix rank hang cua ma tran.",
        "Matrix eigenvector vector nguyen cua ma tran.",
        "Matrix eigenvalue gia tri nguyen cua ma tran.",
        "Matrix decomposition tich phan ma tran.",
        "SVD tich phan gia tri don.",
        "QR tich phan ma tran.",
        "LU tich phan ma tran.",
        "Cholesky tich phan ma tran duong cheo.",
        "Eigendecomposition tich phan ma tran.",
        "Jordan tich phan ma tran.",
        "Singular value gia tri don cua ma tran.",
        "Condition number so dieu kien cua ma tran.",
        "Norm cua ma tran.",
        "Frobenius norm cua ma tran.",
        "Spectral norm cua ma tran.",
        "Nuclear norm cua ma tran.",
        "L1 norm cua vector.",
        "L2 norm cua vector.",
        "Linf norm cua vector.",
        "L0 norm cua vector.",
        "Manhattan distance khoang cach Manhattan.",
        "Euclidean distance khoang cach Euclidean.",
        "Cosine distance khoang cach Cosine.",
        "Jaccard distance khoang cach Jaccard.",
        "Hamming distance khoang cach Hamming.",
        "Levenshtein distance khoang chinh sua xau.",
        "Edit distance khoang chinh sua.",
        "Dynamic programming lap trinh dong bo.",
        "Greedy thuat toan tham lam.",
        "Divide and conquer chia de tri.",
        "Backtracking quay lui.",
        "Branch and bound chi dan va gioi han.",
        "Simulated annealing lam nguoi nguoi.",
        "Genetic algorithm thuat toan di trang sinh.",
        "Particle swarm optimization thuat toan boi tro.",
        "Ant colony optimization thuat toan oc con.",
        "Bee colony optimization thuat toan ong.",
        "Firefly algorithm thuat toan sinh fond.",
        "Bat algorithm thuat toan dong co.",
        "Wolf algorithm thuat toan s福.",
        "Whale optimization algorithm thuat toan ca voi.",
        "Salp swarm algorithm thuat toan ca voi.",
        "Moth flame optimization thuat toan mo.",
        "Grey wolf optimizer thuat toan den.",
        "Harris hawks optimization thuat toan oc.",
        "Slime mould algorithm thuat toan nam.",
        "Archimedes optimization algorithm thuat toan.",
        "Henry gas solubility optimization thuat toan.",
        "Water cycle algorithm thuat toan nuoc.",
        "Evaporation rate optimization thuat toan.",
        "River formation algorithm thuat toan song.",
        "Astronomical optimization algorithm thuat toan.",
        "Big bang optimization algorithm thuat toan.",
        "Black hole optimization algorithm thuat toan.",
        "Galaxy based optimization algorithm thuat toan.",
        "Solar system optimization algorithm thuat toan.",
        "Planetary system optimization algorithm thuat toan.",
        "Atom search optimization algorithm thuat toan.",
        "Electromagnetism optimization algorithm thuat toan.",
        "Magnetic optimization algorithm thuat toan.",
        "Central force optimization algorithm thuat toan.",
        "Gravitational search algorithm thuat toan luc tinh.",
        "Gravity search algorithm thuat toan luc tinh.",
        "Multi-objective optimization thuat toan nhieu muc tieu.",
        "Single-objective optimization thuat toan mot muc tieu.",
        "Constrained optimization thuat toan co rang buoc.",
        "Unconstrained optimization thuat toan khong rang buoc.",
        "Nonlinear optimization thuat toan phi tuyen tinh.",
        "Linear optimization thuat toan tuyen tinh.",
        "Quadratic optimization thuat toan bac hai.",
        "Integer optimization thuat toan so nguyen.",
        "Mixed integer optimization thuat toan so nguyentron.",
        "Combinatorial optimization thuat toan hop chap.",
        "Continuous optimization thuat toan lien tuc.",
        "Discrete optimization thuat toan rong.",
        "Stochastic optimization thuat toan ngau nhien.",
        "Deterministic optimization thuat toan dinh dinh.",
        "Global optimization tim toi uu toan cuc.",
        "Local optimization tim toi uu cuc bo.",
        "Convex optimization thuat toan ham luan.",
        "Non-convex optimization thuat toan khong luan.",
        "Smooth optimization thuat toan ham tuong tac.",
        "Nonsmooth optimization thuat toan khong tuong tac.",
        "Lipschitz optimization thuat toan dam bao toc toc.",
        "Strongly convex optimization ham luan manh.",
        "Weakly convex optimization ham luan yeu.",
        "Quasi-convex optimization ham khong luan.",
        "Pseudoconvex optimization ham khong luan.",
        "Invex optimization ham khong luan.",
        "Semidefinite optimization ma tran duong.",
        "Second-order cone optimization ma tran non.",
        "Robust optimization chong giao tiep.",
        "Stochastic programming lap trinh ngau nhien.",
        "Robust optimization chong giao tiep.",
        "Multi-stage optimization nhieu giai doan.",
        "Dynamic programming lap trinh dong bo.",
        "Linear programming lap trinh tuyen tinh.",
        "Integer programming lap trinh so nguyen.",
        "Quadratic programming lap trinh bac hai.",
        "Second-order cone programming lap trinh ma tran non.",
        "Semidefinite programming lap trinh ma tran duong.",
        "Geometric programming lap trinh co hinh.",
        "Fractional programming lap trinh phan so.",
        "Bilevel programming lap trinh nhieu cap do.",
        "Inverse programming lap trinh nghich dao.",
        "Metaheuristic optimization thuat toan meta.",
        "Nature-inspired optimization thuat toan toi tu nhien.",
        "Bio-inspired optimization thuat toan toi tu sinh hoc.",
        "Physics-inspired optimization thuat toan toi tu vat ly.",
        "Swarm intelligence toc do hoa con.",
        "Ant colony optimization thuat toan oc con.",
        "Bee algorithm thuat toan ong.",
        "Firefly algorithm thuat toan sinh fond.",
        "Cuckoo search algorithm thuat toan cuckoo.",
        "Bat algorithm thuat toan dong co.",
        "Wolf algorithm thuat toan s福.",
        "Whale optimization algorithm thuat toan ca voi.",
        "Salp swarm algorithm thuat toan ca voi.",
        "Moth flame optimization thuat toan mo.",
        "Grey wolf optimizer thuat toan den.",
        "Harris hawks optimization thuat toan oc.",
        "Slime mould algorithm thuat toan nam.",
        "Archimedes optimization algorithm thuat toan.",
        "Henry gas solubility optimization thuat toan.",
        "Water cycle algorithm thuat toan nuoc.",
        "Evaporation rate optimization thuat toan.",
        "River formation algorithm thuat toan song.",
        "Astronomical optimization algorithm thuat toan.",
        "Big bang optimization algorithm thuat toan.",
        "Black hole optimization algorithm thuat toan.",
        "Galaxy based optimization algorithm thuat toan.",
        "Solar system optimization algorithm thuat toan.",
        "Planetary system optimization algorithm thuat toan.",
        "Atom search optimization algorithm thuat toan.",
        "Electromagnetism optimization algorithm thuat toan.",
        "Magnetic optimization algorithm thuat toan.",
        "Central force optimization algorithm thuat toan.",
        "Gravitational search algorithm thuat toan luc tinh.",
        "Gravity search algorithm thuat toan luc tinh.",
    ]


def train_llm(
    config_path="configs/model_config.json",
    text_path=None,
    epochs=200,
    lr=0.005,
    batch_size=16,
    seq_len=64,
    save_path="models/llm.npz",
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

    text = load_text(text_path)
    if text is None:
        text = "\n".join(default_texts())

    tokenizer = CharTokenizer()
    tokenizer.fit([text], min_freq=1, max_vocab=llm_cfg["vocab_size"])
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer da luu: {tokenizer_path} (vocab_size={tokenizer.vocab_size})")

    ids = tokenizer.encode(text)
    ids = [min(i, vocab_size - 1) for i in ids]
    ids = np.array(ids, dtype=np.int64)
    total_tokens = len(ids)
    print(f"Tong so token: {total_tokens}, seq_len: {seq_len}")

    data = []
    step = max(1, seq_len // 2)
    for i in range(0, total_tokens - seq_len, step):
        chunk = ids[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            break
        data.append((chunk[:-1], chunk[1:]))
    if len(data) == 0:
        print("Khong du du lieu de huon luyen.")
        return

    X = np.array([d[0] for d in data], dtype=np.int64)
    Y = np.array([d[1] for d in data], dtype=np.int64)
    print(f"So mau huan luyen: {len(X)}")

    model = NanoLLMTorch(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Thiet bet: {device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    best_loss = float("inf")
    for epoch in range(epochs):
        perm = np.random.permutation(len(X))
        epoch_loss = 0.0
        nb = 0
        for i in range(0, len(X), batch_size):
            idxb = perm[i:i + batch_size]
            xb = torch.tensor(X[idxb], dtype=torch.long, device=device)
            yb = torch.tensor(Y[idxb], dtype=torch.long, device=device)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            nb += 1
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss/max(1,nb):.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model.save_npz(save_path)
            print(f"  [best] Epoch {epoch+1} - loss: {epoch_loss/max(1,nb):.4f} (best={best_loss:.4f})")

    if best_loss == float("inf"):
        model.save_npz(save_path)
    print(f"Huon luyen LLM da hoan thanh. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train_llm()
