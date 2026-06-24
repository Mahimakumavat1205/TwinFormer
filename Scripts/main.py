# I3Informer v2 — Temperature Forecasting
import random, math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed(42)

df = pd.read_csv("temperatures.csv")
df.columns = df.columns.str.strip()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

target_col = 'Temperature'
if target_col not in df.columns:
    raise KeyError(f"Column '{target_col}' not found. Available: {list(df.columns)}")

# Cyclical time features
if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['hour_sin'] = np.sin(2 * np.pi * df['Datetime'].dt.hour / 24).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * df['Datetime'].dt.hour / 24).astype(np.float32)
    df['dow_sin']  = np.sin(2 * np.pi * df['Datetime'].dt.dayofweek / 7).astype(np.float32)
    df['dow_cos']  = np.cos(2 * np.pi * df['Datetime'].dt.dayofweek / 7).astype(np.float32)
    TIME_FEATURES  = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    print("Cyclical time features added")
else:
    TIME_FEATURES = []
    print("No 'Datetime' column — skipping time features")

# Lag features (t-24 and t-48 of temperature)
LAG_STEPS = [24, 48]
for lag in LAG_STEPS:
    df[f'temp_lag{lag}'] = df[target_col].shift(lag)
lag_cols = [f'temp_lag{lag}' for lag in LAG_STEPS]

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Temperature is both the feature and the target
ALL_FEATURES = [target_col] + TIME_FEATURES + lag_cols
print(f"Using {len(ALL_FEATURES)} features: {ALL_FEATURES}")

X_raw = df[ALL_FEATURES].values.astype(np.float32)
y_raw = df[[target_col]].values.astype(np.float32)

# 2. Scaling  
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
print("StandardScaler applied")

def inverse_y(arr):
    return y_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

# 3. Sequence creation
def create_sequences(X, y, seq_len=48, pred_len=96):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN   = 48
PRED_LENS = [96, 120, 336, 720]

# 4. Model 

class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2):
        super().__init__()
        self.k, self.h = k, num_heads
        self.dh = embed_dim // num_heads
        self.qp = nn.Linear(embed_dim, embed_dim)
        self.kp = nn.Linear(embed_dim, embed_dim)
        self.vp = nn.Linear(embed_dim, embed_dim)
        self.op = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh   = self.h, self.dh
        q = self.qp(Q).view(B, L, H, Dh).transpose(1, 2)
        k = self.kp(K).view(B, L, H, Dh).transpose(1, 2)
        v = self.vp(V).view(B, L, H, Dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        ek = min(self.k, L)
        tv, ti = torch.topk(sc, ek, dim=-1)
        m = torch.full_like(sc, float('-inf'))
        m.scatter_(-1, ti, tv)
        a = torch.softmax(m, dim=-1)
        o = torch.matmul(a, v)
        return self.op(o.transpose(1, 2).reshape(B, L, E))


class SparseBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2, dropout=0.15):
        super().__init__()
        self.n1   = nn.LayerNorm(embed_dim)
        self.n2   = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        n = self.n1(x)
        x = x + self.drop(self.attn(n, n, n))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class I3InformerV2(nn.Module):
    def __init__(self, input_dim, embed_dim=128, patch_size=12,
                 pred_len=96, topk=2, seq_len=48, dropout=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.embed      = nn.Linear(input_dim, embed_dim)
        self.pe         = SinusoidalPE(embed_dim, max_len=seq_len+16, dropout=dropout)
        self.local1     = SparseBlock(embed_dim, 4, topk, dropout)
        self.local2     = SparseBlock(embed_dim, 4, topk, dropout)
        self.global1    = SparseBlock(embed_dim, 4, topk, dropout)
        self.global2    = SparseBlock(embed_dim, 4, topk, dropout)
        self.rnn        = nn.GRU(embed_dim, embed_dim, num_layers=2,
                                 batch_first=True, dropout=dropout)
        self.norm  = nn.LayerNorm(embed_dim)
        self.fc1   = nn.Linear(embed_dim, embed_dim * 4)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc3   = nn.Linear(embed_dim * 2, pred_len)
        self.skip  = nn.Linear(embed_dim, pred_len)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.pe(self.embed(x))
        T_trim      = (T // self.patch_size) * self.patch_size
        x           = x[:, :T_trim, :]
        num_patches = T_trim // self.patch_size
        p = x.view(B * num_patches, self.patch_size, -1)
        p = self.local2(self.local1(p))
        p = p[:, -1, :].view(B, num_patches, -1)
        g = self.global2(self.global1(p))
        rnn_out, _ = self.rnn(g)
        ctx = rnn_out.mean(dim=1)
        h  = self.norm(ctx)
        h2 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h2)))
        return self.fc3(h2) + self.skip(h)

# 5. Combined loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)

# 6. LR scheduler
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.opt   = optimizer
        self.wu    = warmup_epochs
        self.total = total_epochs
        self.base  = base_lr
        self.min   = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.wu:
            lr = self.base * e / self.wu
        else:
            progress = (e - self.wu) / (self.total - self.wu)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

def get_lr(opt):
    return opt.param_groups[0]['lr']

# 7. Training loop

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EMBED_DIM  = 128
PATCH_SIZE = 12
TOPK       = 2
MAX_EPOCHS = 100
WARMUP     = 5
PATIENCE   = 10
BASE_LR    = 5e-4

results = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*60}\nPred len: {PRED_LEN}\n{'='*60}")
    set_seed(42)

    if len(X_scaled) < SEQ_LEN + PRED_LEN + 10:
        print("Skipping: insufficient data"); continue

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) == 0:
        print("Skipping: no valid sequences"); continue

    n          = len(X_seq)
    tr         = int(0.70 * n)
    va         = int(0.15 * n)
    X_tr, y_tr = X_seq[:tr],        y_seq[:tr]
    X_va, y_va = X_seq[tr:tr+va],   y_seq[tr:tr+va]
    X_te, y_te = X_seq[tr+va:],     y_seq[tr+va:]
    print(f"Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")

    def mk(X, y, shuf=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32).squeeze(-1)),
            batch_size=32, shuffle=shuf)

    tr_ld = mk(X_tr, y_tr, shuf=True)
    va_ld = mk(X_va, y_va)
    te_ld = mk(X_te, y_te)

    model = I3InformerV2(
        input_dim  = len(ALL_FEATURES),
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP, MAX_EPOCHS, BASE_LR)
    criterion = CombinedLoss(alpha=0.5)

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_ld)

        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train {tr_loss:.5f} | "
                  f"val {va_loss:.5f} | lr {get_lr(optimizer):.6f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state); model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            y_pred.extend(model(xb.to(device)).cpu().numpy())
            y_true.extend(yb.numpy())

    yt = inverse_y(np.array(y_true))
    yp = inverse_y(np.array(y_pred))

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)

    results[PRED_LEN] = dict(MAE=mae, RMSE=rmse, R2=r2,
                              state=best_state, X_te=X_te, y_te=y_te)
    print(f"  PRED_LEN={PRED_LEN} → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

# 8. Summary

print(f"\n{'='*70}")
print("I3INFORMER V2 — TEMPERATURE FORECASTING — TEST SET RESULTS")
print(f"{'='*70}")
print(f"{'Pred len':<10}{'MAE':<16}{'RMSE':<16}{'R²'}")
print("-" * 55)
for pl in PRED_LENS:
    r = results.get(pl)
    if r:
        print(f"{pl:<10}{r['MAE']:<16.4f}{r['RMSE']:<16.4f}{r['R2']:.4f}")
    else:
        print(f"{pl:<10}{'SKIPPED':<16}{'SKIPPED':<16}{'SKIPPED'}")

# 9. Plot

valid = [pl for pl in PRED_LENS if pl in results]
if valid:
    PRED_LEN = valid[0]
    r        = results[PRED_LEN]
    model.load_state_dict(r['state']); model.eval()

    yt_all, yp_all = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(
            TensorDataset(torch.tensor(r['X_te'], dtype=torch.float32),
                          torch.tensor(r['y_te'], dtype=torch.float32).squeeze(-1)),
                batch_size=32):
            yp_all.extend(model(xb.to(device)).cpu().numpy())
            yt_all.extend(yb.numpy())

    yt_inv = inverse_y(np.array(yt_all))
    yp_inv = inverse_y(np.array(yp_all))
    n_plot = min(100, len(yt_inv))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(yt_inv[:n_plot], label='True',      linewidth=2)
    axes[0].plot(yp_inv[:n_plot], label='Predicted', linewidth=2, alpha=0.8)
    axes[0].set_title(f"Temperature Forecast (I3Informer V2) — Pred len {PRED_LEN}",
                      fontsize=14)
    axes[0].set_ylabel("Temperature"); axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    err = yt_inv[:n_plot] - yp_inv[:n_plot]
    axes[1].bar(range(n_plot), err, alpha=0.6, color='steelblue', label='Error')
    axes[1].axhline(0, color='red', linewidth=1)
    axes[1].set_title("Prediction error (true − predicted)", fontsize=12)
    axes[1].set_ylabel("Error"); axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fname = f'temperature_forecast_v2_predlen_{PRED_LEN}.png'
    plt.savefig(fname, dpi=120); plt.close()
    print(f"\nPlot saved as '{fname}'")

print("\n I3Informer v2 — Temperature forecasting completed!")


# In[]:
# v2 —Power Consumption

import random, math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & enrich dataset
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("powerconsumption.csv")
df.columns = df.columns.str.strip()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# D2 — Cyclical time features (requires a Datetime column)
if 'Datetime' in df.columns:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['hour_sin']    = np.sin(2 * np.pi * df['Datetime'].dt.hour / 24).astype(np.float32)
    df['hour_cos']    = np.cos(2 * np.pi * df['Datetime'].dt.hour / 24).astype(np.float32)
    df['dow_sin']     = np.sin(2 * np.pi * df['Datetime'].dt.dayofweek / 7).astype(np.float32)
    df['dow_cos']     = np.cos(2 * np.pi * df['Datetime'].dt.dayofweek / 7).astype(np.float32)
    TIME_FEATURES     = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    print("✅ Cyclical time features added")
else:
    TIME_FEATURES = []
    print("⚠  No 'Datetime' column found — skipping time features")

possible_features = ['Temperature','Humidity','WindSpeed',
                     'GeneralDiffuseFlows','DiffuseFlows']
base_features = [c for c in possible_features if c in df.columns]
if not base_features:
    raise ValueError("No feature columns found.")

target_col = 'PowerConsumption_Zone1'
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found.")

# D3 — Lag features (t-24 and t-48 of the target, appended as extra input dims)
LAG_STEPS = [24, 48]
for lag in LAG_STEPS:
    col = f'target_lag{lag}'
    df[col] = df[target_col].shift(lag)
lag_cols = [f'target_lag{lag}' for lag in LAG_STEPS]

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

ALL_FEATURES = base_features + TIME_FEATURES + lag_cols
print(f"Using {len(ALL_FEATURES)} features: {ALL_FEATURES}")

X_raw = df[ALL_FEATURES].values.astype(np.float32)
y_raw = df[[target_col]].values.astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Scaling  (D1 — StandardScaler)
# ═══════════════════════════════════════════════════════════════════════════════
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
print("✅ StandardScaler applied to features and target")

def inverse_y(arr):
    return y_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sequence creation
# ═══════════════════════════════════════════════════════════════════════════════
def create_sequences(X, y, seq_len=48, pred_len=96):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN   = 48
PRED_LENS = [96, 120, 336, 720]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Model definition — I3Informer v2
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2):
        super().__init__()
        self.k, self.h = k, num_heads
        self.dh = embed_dim // num_heads
        self.qp = nn.Linear(embed_dim, embed_dim)
        self.kp = nn.Linear(embed_dim, embed_dim)
        self.vp = nn.Linear(embed_dim, embed_dim)
        self.op = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh   = self.h, self.dh
        q = self.qp(Q).view(B, L, H, Dh).transpose(1, 2)
        k = self.kp(K).view(B, L, H, Dh).transpose(1, 2)
        v = self.vp(V).view(B, L, H, Dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        ek = min(self.k, L)
        tv, ti = torch.topk(sc, ek, dim=-1)
        m = torch.full_like(sc, float('-inf'))
        m.scatter_(-1, ti, tv)
        a = torch.softmax(m, dim=-1)
        o = torch.matmul(a, v)
        return self.op(o.transpose(1, 2).reshape(B, L, E))


class SparseBlock(nn.Module):
    """Pre-norm sparse attention block with dropout."""
    def __init__(self, embed_dim, num_heads=4, k=2, dropout=0.15):
        super().__init__()
        self.n1   = nn.LayerNorm(embed_dim)
        self.n2   = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        n = self.n1(x)
        x = x + self.drop(self.attn(n, n, n))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class I3InformerV2(nn.Module):
    """
    I3Informer v2:
      - 2 stacked local SparseBlocks (within patches)
      - 2 stacked global SparseBlocks (across patch tokens)
      - 2-layer GRU, full output mean-pooled
      - Deep residual projection head: E→4E→2E→pred_len
      - embed_dim=128
    """
    def __init__(self, input_dim, embed_dim=128, patch_size=12,
                 pred_len=96, topk=2, seq_len=48, dropout=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.embed      = nn.Linear(input_dim, embed_dim)
        self.pe         = SinusoidalPE(embed_dim, max_len=seq_len+16, dropout=dropout)

        # M1 — stacked local blocks (×2)
        self.local1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.local2 = SparseBlock(embed_dim, 4, topk, dropout)

        # M1 — stacked global blocks (×2)
        self.global1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.global2 = SparseBlock(embed_dim, 4, topk, dropout)

        # GRU (2 layers)
        self.rnn = nn.GRU(embed_dim, embed_dim, num_layers=2,
                          batch_first=True, dropout=dropout)

        # M3 — deep residual decoder head
        self.norm  = nn.LayerNorm(embed_dim)
        self.fc1   = nn.Linear(embed_dim, embed_dim * 4)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc3   = nn.Linear(embed_dim * 2, pred_len)
        # skip connection: project embed → pred_len directly
        self.skip  = nn.Linear(embed_dim, pred_len)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.pe(self.embed(x))

        T_trim      = (T // self.patch_size) * self.patch_size
        x           = x[:, :T_trim, :]
        num_patches = T_trim // self.patch_size

        # Local: 2 stacked blocks within each patch
        p = x.view(B * num_patches, self.patch_size, -1)
        p = self.local2(self.local1(p))
        p = p[:, -1, :].view(B, num_patches, -1)   # last-token aggregation

        # Global: 2 stacked blocks across patch tokens
        g = self.global2(self.global1(p))

        # GRU → mean pool
        rnn_out, _ = self.rnn(g)
        ctx        = rnn_out.mean(dim=1)             # (B, E)

        # Deep residual head
        h  = self.norm(ctx)
        h2 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h2)))
        return self.fc3(h2) + self.skip(h)           # residual skip


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Combined loss  (L1)
# ═══════════════════════════════════════════════════════════════════════════════
class CombinedLoss(nn.Module):
    """0.5 × MAE + 0.5 × MSE — balances spike-robustness and scale sensitivity."""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LR scheduler helpers  (T1)
# ═══════════════════════════════════════════════════════════════════════════════
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

class WarmupCosineScheduler:
    """Linear warm-up for `warmup_epochs`, then CosineAnnealingLR."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.opt    = optimizer
        self.wu     = warmup_epochs
        self.total  = total_epochs
        self.base   = base_lr
        self.min    = min_lr
        self.epoch  = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.wu:
            lr = self.base * e / self.wu
        else:
            progress = (e - self.wu) / (self.total - self.wu)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Training loop
# ═══════════════════════════════════════════════════════════════════════════════
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EMBED_DIM  = 128
PATCH_SIZE = 12
TOPK       = 2
MAX_EPOCHS = 100
WARMUP     = 5
PATIENCE   = 10        # T2 early stopping
BASE_LR    = 5e-4

results = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*60}\nPred len: {PRED_LEN}\n{'='*60}")
    set_seed(42)

    if len(X_scaled) < SEQ_LEN + PRED_LEN + 10:
        print(f"❌ Skipping: insufficient data"); continue

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) == 0:
        print(f"❌ Skipping: no valid sequences"); continue

    n          = len(X_seq)
    tr         = int(0.70 * n)
    va         = int(0.15 * n)
    X_tr, y_tr = X_seq[:tr],        y_seq[:tr]
    X_va, y_va = X_seq[tr:tr+va],   y_seq[tr:tr+va]
    X_te, y_te = X_seq[tr+va:],     y_seq[tr+va:]
    print(f"Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")

    def mk(X, y, shuf=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32).squeeze(-1)),
            batch_size=32, shuffle=shuf)

    tr_ld = mk(X_tr, y_tr, shuf=True)   # T3 shuffle=True
    va_ld = mk(X_va, y_va)
    te_ld = mk(X_te, y_te)

    model = I3InformerV2(
        input_dim  = len(ALL_FEATURES),
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler  = WarmupCosineScheduler(optimizer, WARMUP, MAX_EPOCHS, BASE_LR)
    criterion  = CombinedLoss(alpha=0.5)           # L1

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train(); tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        # ── validate ───────────────────────────────────────────────────────────
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_ld)

        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train {tr_loss:.5f} | val {va_loss:.5f} | lr {get_lr(optimizer):.6f}")

        if no_improve >= PATIENCE:          # T2 early stopping
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    # ── evaluate ───────────────────────────────────────────────────────────────
    model.load_state_dict(best_state); model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            y_pred.extend(model(xb.to(device)).cpu().numpy())
            y_true.extend(yb.numpy())

    yt = inverse_y(np.array(y_true))
    yp = inverse_y(np.array(y_pred))

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)

    results[PRED_LEN] = dict(MAE=mae, RMSE=rmse, R2=r2,
                              state=best_state, X_te=X_te, y_te=y_te)
    print(f"  PRED_LEN={PRED_LEN} → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("I3INFORMER V2 — TEST SET RESULTS")
print(f"{'='*70}")
print(f"{'Pred len':<10}{'MAE':<16}{'RMSE':<16}{'R²'}")
print("-" * 60)
for pl in PRED_LENS:
    r = results.get(pl)
    if r:
        print(f"{pl:<10}{r['MAE']:<16.4f}{r['RMSE']:<16.4f}{r['R2']:.4f}")
    else:
        print(f"{pl:<10}{'SKIPPED':<16}{'SKIPPED':<16}{'SKIPPED'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Plot
# ═══════════════════════════════════════════════════════════════════════════════
valid = [pl for pl in PRED_LENS if pl in results]
if valid:
    PRED_LEN = valid[0]
    r        = results[PRED_LEN]
    model.load_state_dict(r['state']); model.eval()

    yt_all, yp_all = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(
            TensorDataset(torch.tensor(r['X_te'], dtype=torch.float32),
                          torch.tensor(r['y_te'], dtype=torch.float32).squeeze(-1)),
                batch_size=32):
            yp_all.extend(model(xb.to(device)).cpu().numpy())
            yt_all.extend(yb.numpy())

    yt_inv = inverse_y(np.array(yt_all))
    yp_inv = inverse_y(np.array(yp_all))
    n_plot = min(100, len(yt_inv))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(yt_inv[:n_plot], label='True',      linewidth=2)
    axes[0].plot(yp_inv[:n_plot], label='Predicted', linewidth=2, alpha=0.8)
    axes[0].set_title(f"I3Informer V2 — Pred len {PRED_LEN}", fontsize=14)
    axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.5)

    err = yt_inv[:n_plot] - yp_inv[:n_plot]
    axes[1].bar(range(n_plot), err, alpha=0.6, color='steelblue', label='Error')
    axes[1].axhline(0, color='red', linewidth=1)
    axes[1].set_title("Prediction error (true − predicted)", fontsize=12)
    axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fname = f'i3informer_v2_predlen_{PRED_LEN}.png'
    plt.savefig(fname, dpi=120); plt.close()
    print(f"\nPlot saved as '{fname}'")

print("\n✅ I3Informer v2 completed!")


# In[]:
# I3InformerV2 — Weather Dataset

import warnings
warnings.filterwarnings('ignore')

import random, math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'reset'):
    torch._dynamo.reset()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 1. Data loading ────────────────────────────────────────────────────────────

df = pd.read_csv("weather_utf8.csv")
df['Date Time'] = pd.to_datetime(df['Date Time'], dayfirst=True, errors='coerce')
df.set_index('Date Time', inplace=True)

feature_cols = [
    'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
    'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m²)',
    'PAR (µmol/m²/s)', 'max. PAR (µmol/m²/s)', 'Tlog (degC)', 'CO2 (ppm)'
]

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=feature_cols, inplace=True)
df.reset_index(inplace=True)   # bring 'Date Time' back as a column for time features

# ── 2. Cyclical time features (D2 — same as power consumption) ────────────────
# Weather datetime index is 10-minute intervals; hour and day-of-week still apply.

df['hour_sin'] = np.sin(2 * np.pi * df['Date Time'].dt.hour / 24).astype(np.float32)
df['hour_cos'] = np.cos(2 * np.pi * df['Date Time'].dt.hour / 24).astype(np.float32)
df['dow_sin']  = np.sin(2 * np.pi * df['Date Time'].dt.dayofweek / 7).astype(np.float32)
df['dow_cos']  = np.cos(2 * np.pi * df['Date Time'].dt.dayofweek / 7).astype(np.float32)
TIME_FEATURES  = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
print("Cyclical time features added:", TIME_FEATURES)

# ── 3. Lag features (D3 — same lag steps as power consumption) ────────────────
# Power consumption lags the single target (zone1) by 24 and 48 steps.
# Weather is multivariate: we lag ALL 21 features by the same steps so the
# model sees "what every variable looked like 24/48 steps ago" — the direct
# multivariate generalisation of the univariate lag approach.

LAG_STEPS = [24, 48]
lag_cols   = []
for lag in LAG_STEPS:
    for col in feature_cols:
        lag_col = f'{col}_lag{lag}'
        df[lag_col] = df[col].shift(lag)
        lag_cols.append(lag_col)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

ALL_INPUT_COLS = feature_cols + TIME_FEATURES + lag_cols
print(f"Total input features: {len(ALL_INPUT_COLS)}")
print(f"  Base weather : {len(feature_cols)}")
print(f"  Time cyclical: {len(TIME_FEATURES)}")
print(f"  Lag features : {len(lag_cols)}")

INPUT_DIM  = len(ALL_INPUT_COLS)   # model input width
TARGET_DIM = len(feature_cols)     # model output width (predict the 21 raw features)

X_raw = df[ALL_INPUT_COLS].values.astype(np.float32)   # (N, INPUT_DIM)
y_raw = df[feature_cols].values.astype(np.float32)     # (N, TARGET_DIM)

# ── 4. Scaling — StandardScaler on X and Y (same as power consumption) ────────

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
print("StandardScaler applied to X and Y")

def inverse_y(arr_2d):
    """Inverse-transform scaled predictions/targets back to original space.
    arr_2d : (N, TARGET_DIM)  →  (N, TARGET_DIM)
    """
    return y_scaler.inverse_transform(arr_2d)

# ── 5. Train / Val / Test split ───────────────────────────────────────────────

n         = len(X_scaled)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

X_train_full = X_scaled[:train_end]
y_train_full = y_scaled[:train_end]
X_val_full   = X_scaled[train_end:val_end]
y_val_full   = y_scaled[train_end:val_end]
X_test_full  = X_scaled[val_end:]
y_test_full  = y_scaled[val_end:]
print(f"Split — Train: {len(X_train_full)}, Val: {len(X_val_full)}, Test: {len(X_test_full)}")

# ── 6. Sequence creation ───────────────────────────────────────────────────────
# X sequences come from the full feature matrix (base + time + lags).
# y sequences come from the target-only matrix (21 raw features).

def create_sequences(X, y, seq_len, pred_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + pred_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN   = 48
PRED_LENS = [96, 120, 336, 720]


# ── 7. Model ───────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2):
        super().__init__()
        self.k, self.h = k, num_heads
        self.dh = embed_dim // num_heads
        self.qp = nn.Linear(embed_dim, embed_dim)
        self.kp = nn.Linear(embed_dim, embed_dim)
        self.vp = nn.Linear(embed_dim, embed_dim)
        self.op = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh   = self.h, self.dh
        q = self.qp(Q).view(B, L, H, Dh).transpose(1, 2)
        k = self.kp(K).view(B, L, H, Dh).transpose(1, 2)
        v = self.vp(V).view(B, L, H, Dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        ek = min(self.k, L)
        tv, ti = torch.topk(sc, ek, dim=-1)
        m = torch.full_like(sc, float('-inf'))
        m.scatter_(-1, ti, tv)
        a = torch.softmax(m, dim=-1)
        o = torch.matmul(a, v)
        return self.op(o.transpose(1, 2).reshape(B, L, E))


class SparseBlock(nn.Module):
    """Pre-norm sparse attention block — identical to I3InformerV2."""
    def __init__(self, embed_dim, num_heads=4, k=2, dropout=0.15):
        super().__init__()
        self.n1   = nn.LayerNorm(embed_dim)
        self.n2   = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        n = self.n1(x)
        x = x + self.drop(self.attn(n, n, n))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class I3InformerV2Weather(nn.Module):
    """
    I3InformerV2 — full port for multivariate weather forecasting.

    Input  : (B, seq_len, INPUT_DIM)   where INPUT_DIM = 21 base
                                        + 4 cyclical + 42 lag = 67
    Output : (B, pred_len, TARGET_DIM) where TARGET_DIM = 21 (raw features)

    Architecture mirrors power consumption model exactly:
      - embed_dim = 128
      - 2 stacked local  SparseBlocks (within patches)
      - 2 stacked global SparseBlocks (across patch tokens)
      - 2-layer GRU, full output mean-pooled → context vector
      - Deep residual head: E → 4E → 2E → pred_len*TARGET_DIM
      - Skip connection:    E → pred_len*TARGET_DIM
    """
    def __init__(self, input_dim, target_dim=21, embed_dim=128, patch_size=12,
                 pred_len=96, topk=2, seq_len=48, dropout=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.pred_len   = pred_len
        self.target_dim = target_dim
        out_size        = pred_len * target_dim

        self.embed = nn.Linear(input_dim, embed_dim)
        self.pe    = SinusoidalPE(embed_dim, max_len=seq_len + 16, dropout=dropout)

        # 2 stacked local blocks (within each patch)
        self.local1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.local2 = SparseBlock(embed_dim, 4, topk, dropout)

        # 2 stacked global blocks (across patch tokens)
        self.global1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.global2 = SparseBlock(embed_dim, 4, topk, dropout)

        # 2-layer GRU → mean pool
        self.rnn = nn.GRU(embed_dim, embed_dim, num_layers=2,
                          batch_first=True, dropout=dropout)

        # Deep residual decoder head: E → 4E → 2E → out_size
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1  = nn.Linear(embed_dim, embed_dim * 4)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc3  = nn.Linear(embed_dim * 2, out_size)
        # Skip connection: E → out_size
        self.skip = nn.Linear(embed_dim, out_size)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.pe(self.embed(x))                       # (B, T, E)

        T_trim      = (T // self.patch_size) * self.patch_size
        x           = x[:, :T_trim, :]
        num_patches = T_trim // self.patch_size

        # Local: 2 stacked blocks within each patch
        p = x.view(B * num_patches, self.patch_size, -1)
        p = self.local2(self.local1(p))
        p = p[:, -1, :].view(B, num_patches, -1)         # last-token aggregation

        # Global: 2 stacked blocks across patch tokens
        g = self.global2(self.global1(p))                # (B, P, E)

        # 2-layer GRU → mean pool
        rnn_out, _ = self.rnn(g)
        ctx        = rnn_out.mean(dim=1)                  # (B, E)

        # Deep residual head
        h  = self.norm(ctx)
        h2 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h2)))
        out = self.fc3(h2) + self.skip(h)                # residual skip

        return out.view(B, self.pred_len, self.target_dim)  # (B, pred_len, F)


# ── 8. Combined loss (0.5×MAE + 0.5×MSE — same as power consumption) ─────────

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)


# ── 9. Warmup-cosine LR scheduler (same as power consumption) ─────────────────

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

class WarmupCosineScheduler:
    """Linear warm-up then cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.opt   = optimizer
        self.wu    = warmup_epochs
        self.total = total_epochs
        self.base  = base_lr
        self.min   = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.wu:
            lr = self.base * e / self.wu
        else:
            progress = (e - self.wu) / (self.total - self.wu)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr


# ── 10. Hyperparameters (same as power consumption) ───────────────────────────

EMBED_DIM  = 128
PATCH_SIZE = 12
TOPK       = 2
MAX_EPOCHS = 100
WARMUP     = 5
PATIENCE   = 10
BASE_LR    = 5e-4

results = {}

# ── 11. Training loop ─────────────────────────────────────────────────────────

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*60}\nPred len: {PRED_LEN}\n{'='*60}")
    set_seed(42)

    X_tr, y_tr = create_sequences(X_train_full, y_train_full, SEQ_LEN, PRED_LEN)
    X_va, y_va = create_sequences(X_val_full,   y_val_full,   SEQ_LEN, PRED_LEN)
    X_te, y_te = create_sequences(X_test_full,  y_test_full,  SEQ_LEN, PRED_LEN)

    if any(len(s) == 0 for s in [X_tr, X_va, X_te]):
        print(f"Skipping PRED_LEN={PRED_LEN}: insufficient data")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None,
                             'state': None, 'X_te': None, 'y_te': None}
        continue

    print(f"Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")

    def make_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32)),
            batch_size=32, shuffle=shuffle)

    tr_ld = make_loader(X_tr, y_tr, shuffle=True)
    va_ld = make_loader(X_va, y_va)
    te_ld = make_loader(X_te, y_te)

    model = I3InformerV2Weather(
        input_dim  = INPUT_DIM,
        target_dim = TARGET_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP, MAX_EPOCHS, BASE_LR)
    criterion = CombinedLoss(alpha=0.5)

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train(); tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        # ── validate ───────────────────────────────────────────────────────────
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_ld)

        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train {tr_loss:.5f} "
                  f"| val {va_loss:.5f} | lr {get_lr(optimizer):.6f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    # ── evaluate ───────────────────────────────────────────────────────────────
    model.load_state_dict(best_state); model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            y_pred_list.append(model(xb.to(device)).cpu().numpy())
            y_true_list.append(yb.numpy())

    y_true = np.concatenate(y_true_list, axis=0)   # (N, pred_len, TARGET_DIM)
    y_pred = np.concatenate(y_pred_list, axis=0)

    N, H, F = y_true.shape
    # Inverse-transform: reshape to (N*H, F), unscale, reshape back
    y_true_inv = inverse_y(y_true.reshape(-1, F))
    y_pred_inv = inverse_y(y_pred.reshape(-1, F))

    mae  = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2   = r2_score(y_true_inv, y_pred_inv)

    results[PRED_LEN] = dict(MAE=mae, RMSE=rmse, R2=r2,
                              state=best_state, X_te=X_te, y_te=y_te)
    print(f"  PRED_LEN={PRED_LEN} → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")


# ── 12. Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("I3INFORMER V2 — WEATHER FORECASTING (TEST SET)")
print(f"{'='*70}")
print(f"{'Pred len':<12}{'MAE':<18}{'RMSE':<18}{'R²'}")
print("-" * 60)
for pl in PRED_LENS:
    r = results.get(pl)
    if r and r['MAE'] is not None:
        print(f"{pl:<12}{r['MAE']:<18.4f}{r['RMSE']:<18.4f}{r['R2']:.4f}")
    else:
        print(f"{pl:<12}{'SKIPPED':<18}{'SKIPPED':<18}{'SKIPPED'}")


# ── 13. Plots ─────────────────────────────────────────────────────────────────

valid_lengths = [pl for pl in PRED_LENS if results.get(pl, {}).get('MAE') is not None]

for PRED_LEN in valid_lengths:
    r = results[PRED_LEN]

    model_plot = I3InformerV2Weather(
        input_dim  = INPUT_DIM,
        target_dim = TARGET_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)
    model_plot.load_state_dict(r['state'])
    model_plot.eval()

    X_last = torch.tensor(r['X_te'][-1:], dtype=torch.float32).to(device)
    y_last_true_scaled = r['y_te'][-1]                      # (pred_len, TARGET_DIM)

    with torch.no_grad():
        y_last_pred_scaled = model_plot(X_last).cpu().numpy()[0]  # (pred_len, TARGET_DIM)

    # Inverse-transform and plot feature 0 (p mbar)
    y_true_plot = inverse_y(y_last_true_scaled)[:, 0]
    y_pred_plot = inverse_y(y_last_pred_scaled)[:, 0]

    # Context: last 24 steps of X, feature 0 (first of the base features in X)
    # X sequences store scaled base+time+lag; feature 0 in X is 'p (mbar)' scaled
    context_scaled_x = r['X_te'][-1, -24:, 0:1]            # (24, 1)
    # Reconstruct original scale: x_scaler was fit on ALL_INPUT_COLS, col 0 = 'p (mbar)'
    context_plot = (context_scaled_x[:, 0] * x_scaler.scale_[0]) + x_scaler.mean_[0]

    time_context = np.arange(SEQ_LEN - 24, SEQ_LEN)
    time_pred    = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

    plt.figure(figsize=(14, 5))
    plt.plot(time_context, context_plot, color='lightgray',
             label='Context (last 24 steps)', linewidth=1)
    plt.plot(time_pred, y_true_plot, 'b-',
             label=f'Actual — {feature_cols[0]}', linewidth=2)
    plt.plot(time_pred, y_pred_plot, 'r--',
             label=f'Predicted — {feature_cols[0]}', linewidth=2)
    plt.axvline(x=SEQ_LEN, color='k', linestyle=':', alpha=0.7,
                label='Prediction Start')
    plt.title(f'Weather — I3InformerV2 | Pred Length = {PRED_LEN}  |  '
              f'Feature: {feature_cols[0]}', fontsize=13)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel(feature_cols[0], fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f'i3informer_v2_weather_predlen_{PRED_LEN}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: '{filename}'")

print("\nI3InformerV2 Weather (full port) — completed!")


# In[]:
# I3InformerV2 — IDEA Stock Dataset (Full port from Power Consumption)
# Includes: cyclical time features, lag features, StandardScaler on X and Y

import warnings
warnings.filterwarnings('ignore')

import random, math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=10):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 1. Data loading ────────────────────────────────────────────────────────────

df = pd.read_csv("IDEA.csv")
df.columns = df.columns.str.strip()
df["Date"]  = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(subset=["Close", "Date"], inplace=True)
df.reset_index(drop=True, inplace=True)

assert np.all(df["Close"].values > 0), "Close prices must be positive for log transformation"
print(f"Loaded {len(df)} rows from IDEA.csv")


# ── 2. Cyclical time features (D2 — same as power consumption) ────────────────
# Stock data is daily; hour_sin/cos collapse to zero (always hour=0).
# We keep day-of-week (Monday..Friday cycle) and add day-of-month / month
# cyclicals instead of hour, matching the daily periodicity of stock data.

df['dow_sin']  = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7).astype(np.float32)
df['dow_cos']  = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7).astype(np.float32)
df['dom_sin']  = np.sin(2 * np.pi * df['Date'].dt.day / 31).astype(np.float32)
df['dom_cos']  = np.cos(2 * np.pi * df['Date'].dt.day / 31).astype(np.float32)
df['mon_sin']  = np.sin(2 * np.pi * df['Date'].dt.month / 12).astype(np.float32)
df['mon_cos']  = np.cos(2 * np.pi * df['Date'].dt.month / 12).astype(np.float32)
TIME_FEATURES  = ['dow_sin', 'dow_cos', 'dom_sin', 'dom_cos', 'mon_sin', 'mon_cos']
print(f"Cyclical time features added: {TIME_FEATURES}")


# ── 3. Lag features (D3 — same lag steps as power consumption) ────────────────
# Power consumption lags the target by 24 and 48 steps (10-min intervals).
# IDEA is daily, so lag-24 = ~1 month ago, lag-48 = ~2 months ago — meaningful
# for stock seasonality.

LAG_STEPS = [24, 48]
lag_cols   = []
for lag in LAG_STEPS:
    col = f'close_lag{lag}'
    df[col] = df['Close'].shift(lag)
    lag_cols.append(col)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Lag features added: {lag_cols}")

ALL_INPUT_COLS = ['Close'] + TIME_FEATURES + lag_cols
print(f"\nTotal input features : {len(ALL_INPUT_COLS)}")
print(f"  Base (Close)       : 1")
print(f"  Time cyclical      : {len(TIME_FEATURES)}")
print(f"  Lag features       : {len(lag_cols)}")

INPUT_DIM = len(ALL_INPUT_COLS)   # model input width  (1 + 6 + 2 = 9)
# Target is still univariate — predict Close price only
TARGET_COL = 'Close'


# ── 4. Scaling — StandardScaler on X and Y (same as power consumption) ────────

X_raw = df[ALL_INPUT_COLS].values.astype(np.float32)   # (N, INPUT_DIM)
y_raw = df[[TARGET_COL]].values.astype(np.float32)     # (N, 1)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
print("StandardScaler applied to X and y")

def inverse_y(arr):
    """Inverse-transform a flat or (N,1) array back to original price scale."""
    return y_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()


# ── 5. Train / Val / Test split ───────────────────────────────────────────────

n         = len(X_scaled)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

X_train_full = X_scaled[:train_end];        y_train_full = y_scaled[:train_end]
X_val_full   = X_scaled[train_end:val_end]; y_val_full   = y_scaled[train_end:val_end]
X_test_full  = X_scaled[val_end:];         y_test_full  = y_scaled[val_end:]
print(f"Split — Train: {len(X_train_full)}, Val: {len(X_val_full)}, Test: {len(X_test_full)}")


# ── 6. Sequence creation ───────────────────────────────────────────────────────

def create_sequences(X, y, seq_len, pred_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + pred_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN   = 48
PRED_LENS = [96, 120, 336, 720]


# ── 7. Model (I3InformerV2, univariate output) ────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2):
        super().__init__()
        self.k, self.h = k, num_heads
        self.dh = embed_dim // num_heads
        self.qp = nn.Linear(embed_dim, embed_dim)
        self.kp = nn.Linear(embed_dim, embed_dim)
        self.vp = nn.Linear(embed_dim, embed_dim)
        self.op = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh   = self.h, self.dh
        q = self.qp(Q).view(B, L, H, Dh).transpose(1, 2)
        k = self.kp(K).view(B, L, H, Dh).transpose(1, 2)
        v = self.vp(V).view(B, L, H, Dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        ek = min(self.k, L)
        tv, ti = torch.topk(sc, ek, dim=-1)
        m = torch.full_like(sc, float('-inf'))
        m.scatter_(-1, ti, tv)
        a = torch.softmax(m, dim=-1)
        o = torch.matmul(a, v)
        return self.op(o.transpose(1, 2).reshape(B, L, E))


class SparseBlock(nn.Module):
    """Pre-norm sparse attention block — identical to I3InformerV2."""
    def __init__(self, embed_dim, num_heads=4, k=2, dropout=0.15):
        super().__init__()
        self.n1   = nn.LayerNorm(embed_dim)
        self.n2   = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        n = self.n1(x)
        x = x + self.drop(self.attn(n, n, n))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class I3InformerV2Stock(nn.Module):
    """
    I3InformerV2 — full port for univariate stock forecasting (IDEA Close price).

    Input  : (B, seq_len, INPUT_DIM)  where INPUT_DIM = 9
                                       (1 Close + 6 cyclical + 2 lag)
    Output : (B, pred_len)            univariate — matches power consumption shape

    Architecture mirrors power consumption model exactly:
      - embed_dim = 128
      - 2 stacked local  SparseBlocks (within patches)
      - 2 stacked global SparseBlocks (across patch tokens)
      - 2-layer GRU, full output mean-pooled → context vector
      - Deep residual head: E → 4E → 2E → pred_len
      - Skip connection:    E → pred_len
    """
    def __init__(self, input_dim=9, embed_dim=128, patch_size=12,
                 pred_len=96, topk=2, seq_len=48, dropout=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.pred_len   = pred_len

        self.embed = nn.Linear(input_dim, embed_dim)
        self.pe    = SinusoidalPE(embed_dim, max_len=seq_len + 16, dropout=dropout)

        # 2 stacked local blocks (within each patch)
        self.local1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.local2 = SparseBlock(embed_dim, 4, topk, dropout)

        # 2 stacked global blocks (across patch tokens)
        self.global1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.global2 = SparseBlock(embed_dim, 4, topk, dropout)

        # 2-layer GRU → mean pool
        self.rnn = nn.GRU(embed_dim, embed_dim, num_layers=2,
                          batch_first=True, dropout=dropout)

        # Deep residual decoder head: E → 4E → 2E → pred_len
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1  = nn.Linear(embed_dim, embed_dim * 4)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc3  = nn.Linear(embed_dim * 2, pred_len)
        # Skip connection: E → pred_len
        self.skip = nn.Linear(embed_dim, pred_len)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.pe(self.embed(x))                       # (B, T, E)

        T_trim      = (T // self.patch_size) * self.patch_size
        x           = x[:, :T_trim, :]
        num_patches = T_trim // self.patch_size

        # Local: 2 stacked sparse blocks within each patch
        p = x.view(B * num_patches, self.patch_size, -1)
        p = self.local2(self.local1(p))
        p = p[:, -1, :].view(B, num_patches, -1)         # last-token aggregation

        # Global: 2 stacked sparse blocks across patch tokens
        g = self.global2(self.global1(p))                # (B, P, E)

        # 2-layer GRU → mean pool
        rnn_out, _ = self.rnn(g)
        ctx        = rnn_out.mean(dim=1)                  # (B, E)

        # Deep residual head
        h  = self.norm(ctx)
        h2 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h2)))
        return self.fc3(h2) + self.skip(h)               # (B, pred_len)


# ── 8. Combined loss (0.5×MAE + 0.5×MSE — same as power consumption) ─────────

class CombinedLoss(nn.Module):
    """0.5 × MAE + 0.5 × MSE — balances spike-robustness and scale sensitivity."""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)


# ── 9. Warmup-cosine LR scheduler (same as power consumption) ─────────────────

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

class WarmupCosineScheduler:
    """Linear warm-up for `warmup_epochs`, then cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.opt   = optimizer
        self.wu    = warmup_epochs
        self.total = total_epochs
        self.base  = base_lr
        self.min   = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.wu:
            lr = self.base * e / self.wu
        else:
            progress = (e - self.wu) / (self.total - self.wu)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr


# ── 10. Hyperparameters (same as power consumption) ───────────────────────────

EMBED_DIM  = 128
PATCH_SIZE = 12
TOPK       = 2
MAX_EPOCHS = 100
WARMUP     = 5
PATIENCE   = 10
BASE_LR    = 5e-4

results = {}

# ── 11. Training loop ─────────────────────────────────────────────────────────

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*60}\nPred len: {PRED_LEN}\n{'='*60}")
    set_seed(10)

    if len(X_scaled) < SEQ_LEN + PRED_LEN + 10:
        print(f"Skipping: insufficient data"); continue

    X_tr, y_tr = create_sequences(X_train_full, y_train_full, SEQ_LEN, PRED_LEN)
    X_va, y_va = create_sequences(X_val_full,   y_val_full,   SEQ_LEN, PRED_LEN)
    X_te, y_te = create_sequences(X_test_full,  y_test_full,  SEQ_LEN, PRED_LEN)

    if any(len(s) == 0 for s in [X_tr, X_va, X_te]):
        print(f"Skipping PRED_LEN={PRED_LEN}: no valid sequences"); continue

    print(f"Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")

    def make_loader(X, y, shuffle=False):
        # y shape: (N, pred_len, 1) → squeeze to (N, pred_len) for univariate loss
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32).squeeze(-1)),
            batch_size=32, shuffle=shuffle)

    tr_ld = make_loader(X_tr, y_tr, shuffle=True)
    va_ld = make_loader(X_va, y_va)
    te_ld = make_loader(X_te, y_te)

    model = I3InformerV2Stock(
        input_dim  = INPUT_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP, MAX_EPOCHS, BASE_LR)
    criterion = CombinedLoss(alpha=0.5)

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train(); tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        # ── validate ───────────────────────────────────────────────────────────
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_ld)

        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train {tr_loss:.5f} "
                  f"| val {va_loss:.5f} | lr {get_lr(optimizer):.6f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    # ── evaluate ───────────────────────────────────────────────────────────────
    model.load_state_dict(best_state); model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            y_pred_list.extend(model(xb.to(device)).cpu().numpy())
            y_true_list.extend(yb.numpy())

    yt = inverse_y(np.array(y_true_list))   # back to original ₹ scale
    yp = inverse_y(np.array(y_pred_list))

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))

    results[PRED_LEN] = dict(MAE=mae, RMSE=rmse, R2=r2,
                              state=best_state, X_te=X_te, y_te=y_te)
    print(f"  PRED_LEN={PRED_LEN} → MAE={mae:.4f}  RMSE={rmse:.4f}")


# ── 12. Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("I3INFORMER V2 — IDEA STOCK FORECASTING (TEST SET)")
print(f"{'='*60}")
print(f"{'Pred len':<10}{'MAE':<16}{'RMSE':<16}")
print("-" * 55)
for pl in PRED_LENS:
    r = results.get(pl)
    if r and r['MAE'] is not None:
        print(f"{pl:<10}{r['MAE']:<16.4f}{r['RMSE']:<16.4f}")
    else:
        print(f"{pl:<10}{'SKIPPED':<16}{'SKIPPED':<16}{'SKIPPED'}")


# ── 13. Plots ─────────────────────────────────────────────────────────────────

valid_lengths = [pl for pl in PRED_LENS if results.get(pl, {}).get('MAE') is not None]

for PRED_LEN in valid_lengths:
    r = results[PRED_LEN]

    model_plot = I3InformerV2Stock(
        input_dim  = INPUT_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)
    model_plot.load_state_dict(r['state'])
    model_plot.eval()

    X_last = torch.tensor(r['X_te'][-1:], dtype=torch.float32).to(device)
    y_last_true_scaled = r['y_te'][-1]     # (pred_len, 1)

    with torch.no_grad():
        y_last_pred_scaled = model_plot(X_last).cpu().numpy()  # (1, pred_len)

    y_true_plot = inverse_y(y_last_true_scaled.flatten())
    y_pred_plot = inverse_y(y_last_pred_scaled.flatten())

    # Context: last 24 steps of the input sequence, Close feature (col 0 of X)
    # Inverse-transform using x_scaler column 0 (= Close)
    context_scaled = r['X_te'][-1, -24:, 0]    # (24,)
    context_plot   = (context_scaled * x_scaler.scale_[0]) + x_scaler.mean_[0]

    time_context = np.arange(SEQ_LEN - 24, SEQ_LEN)
    time_pred    = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

    plt.figure(figsize=(14, 5))
    plt.plot(time_context, context_plot, color='lightgray',
             label='Context (last 24 days)', linewidth=1)
    plt.plot(time_pred, y_true_plot, 'b-',
             label='Actual Close Price (₹)', linewidth=2)
    plt.plot(time_pred, y_pred_plot, 'r--',
             label='Predicted Close Price (₹)', linewidth=2)
    plt.axvline(x=SEQ_LEN, color='k', linestyle=':', alpha=0.7,
                label='Prediction Start')
    plt.title(f'IDEA Stock — I3InformerV2 | Pred Length = {PRED_LEN} days',
              fontsize=14)
    plt.xlabel('Day Index', fontsize=12)
    plt.ylabel('Close Price (₹)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f'i3informer_v2_idea_predlen_{PRED_LEN}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: '{filename}'")

print("\nI3InformerV2 IDEA Stock (full port) — completed!")


# In[]:
# I3InformerV2 — Electricity Load Dataset 

import warnings
warnings.filterwarnings('ignore')

import random, math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 1. Data loading ────────────────────────────────────────────────────────────

df = pd.read_csv("Electricity_load.csv")
if "Unnamed: 0" in df.columns:
    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

feature_cols = [c for c in df.columns if c.startswith("MT_")]
print(f"Loaded {len(df)} rows  |  MT_ features: {len(feature_cols)}")

df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=feature_cols, inplace=True)
df.reset_index(drop=True, inplace=True)


# ── 2. Cyclical time features (D2 — same as power consumption) ────────────────
# Electricity load data is hourly; hour and day-of-week are the dominant cycles,
# matching the power consumption model exactly.

df['hour_sin'] = np.sin(2 * np.pi * df['Date'].dt.hour / 24).astype(np.float32)
df['hour_cos'] = np.cos(2 * np.pi * df['Date'].dt.hour / 24).astype(np.float32)
df['dow_sin']  = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7).astype(np.float32)
df['dow_cos']  = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7).astype(np.float32)
TIME_FEATURES  = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
print(f"Cyclical time features added: {TIME_FEATURES}")


# ── 3. Lag features (D3 — same lag steps as power consumption) ────────────────
# Electricity data is hourly: lag-24 = same hour yesterday,
# lag-48 = same hour two days ago — the most informative lags for load patterns.
# All MT_ columns are lagged (multivariate generalisation, same as weather port).

LAG_STEPS = [24, 48]
lag_cols   = []
for lag in LAG_STEPS:
    for col in feature_cols:
        lag_col = f'{col}_lag{lag}'
        df[lag_col] = df[col].shift(lag)
        lag_cols.append(lag_col)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Lag features added: {len(lag_cols)}  "
      f"({len(feature_cols)} MT_ cols × {len(LAG_STEPS)} lags)")

ALL_INPUT_COLS = feature_cols + TIME_FEATURES + lag_cols
INPUT_DIM      = len(ALL_INPUT_COLS)   # model input width
TARGET_DIM     = len(feature_cols)     # model output width (predict all MT_ cols)

print(f"\nTotal input features : {INPUT_DIM}")
print(f"  Base (MT_)         : {len(feature_cols)}")
print(f"  Time cyclical      : {len(TIME_FEATURES)}")
print(f"  Lag features       : {len(lag_cols)}")
print(f"Target features      : {TARGET_DIM}")


# ── 4. Scaling — StandardScaler on X and Y (same as power consumption) ────────

X_raw = df[ALL_INPUT_COLS].values.astype(np.float32)   # (N, INPUT_DIM)
y_raw = df[feature_cols].values.astype(np.float32)     # (N, TARGET_DIM)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
print("StandardScaler applied to X and Y")

def inverse_y(arr_2d):
    """Inverse-transform scaled (N, TARGET_DIM) back to original load scale."""
    return y_scaler.inverse_transform(arr_2d)


# ── 5. Train / Val / Test split ───────────────────────────────────────────────

n         = len(X_scaled)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

X_train_full = X_scaled[:train_end];        y_train_full = y_scaled[:train_end]
X_val_full   = X_scaled[train_end:val_end]; y_val_full   = y_scaled[train_end:val_end]
X_test_full  = X_scaled[val_end:];         y_test_full  = y_scaled[val_end:]
print(f"Split — Train: {len(X_train_full)}, Val: {len(X_val_full)}, Test: {len(X_test_full)}")


# ── 6. Sequence creation ───────────────────────────────────────────────────────

def create_sequences(X, y, seq_len, pred_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + pred_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN   = 48
PRED_LENS = [96, 120, 336, 720]


# ── 7. Model (I3InformerV2, multivariate output) ──────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2):
        super().__init__()
        self.k, self.h = k, num_heads
        self.dh = embed_dim // num_heads
        self.qp = nn.Linear(embed_dim, embed_dim)
        self.kp = nn.Linear(embed_dim, embed_dim)
        self.vp = nn.Linear(embed_dim, embed_dim)
        self.op = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh   = self.h, self.dh
        q = self.qp(Q).view(B, L, H, Dh).transpose(1, 2)
        k = self.kp(K).view(B, L, H, Dh).transpose(1, 2)
        v = self.vp(V).view(B, L, H, Dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        ek = min(self.k, L)
        tv, ti = torch.topk(sc, ek, dim=-1)
        m = torch.full_like(sc, float('-inf'))
        m.scatter_(-1, ti, tv)
        a = torch.softmax(m, dim=-1)
        o = torch.matmul(a, v)
        return self.op(o.transpose(1, 2).reshape(B, L, E))


class SparseBlock(nn.Module):
    """Pre-norm sparse attention block — identical to I3InformerV2."""
    def __init__(self, embed_dim, num_heads=4, k=2, dropout=0.15):
        super().__init__()
        self.n1   = nn.LayerNorm(embed_dim)
        self.n2   = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        n = self.n1(x)
        x = x + self.drop(self.attn(n, n, n))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class I3InformerV2Electricity(nn.Module):
    """
    I3InformerV2 — full port for multivariate electricity load forecasting.

    Input  : (B, seq_len, INPUT_DIM)    INPUT_DIM  = MT_ + 4 cyclical + MT_*2 lags
    Output : (B, pred_len, TARGET_DIM)  TARGET_DIM = len(MT_ columns)

    Architecture mirrors power consumption model exactly:
      - embed_dim = 128
      - 2 stacked local  SparseBlocks (within patches)
      - 2 stacked global SparseBlocks (across patch tokens)
      - 2-layer GRU, full output mean-pooled → context vector
      - Deep residual head: E → 4E → 2E → pred_len * TARGET_DIM
      - Skip connection:    E → pred_len * TARGET_DIM
    """
    def __init__(self, input_dim, target_dim, embed_dim=128, patch_size=12,
                 pred_len=96, topk=2, seq_len=48, dropout=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.pred_len   = pred_len
        self.target_dim = target_dim
        out_size        = pred_len * target_dim

        self.embed = nn.Linear(input_dim, embed_dim)
        self.pe    = SinusoidalPE(embed_dim, max_len=seq_len + 16, dropout=dropout)

        # 2 stacked local blocks (within each patch)
        self.local1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.local2 = SparseBlock(embed_dim, 4, topk, dropout)

        # 2 stacked global blocks (across patch tokens)
        self.global1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.global2 = SparseBlock(embed_dim, 4, topk, dropout)

        # 2-layer GRU → mean pool
        self.rnn = nn.GRU(embed_dim, embed_dim, num_layers=2,
                          batch_first=True, dropout=dropout)

        # Deep residual decoder head: E → 4E → 2E → out_size
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1  = nn.Linear(embed_dim, embed_dim * 4)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc3  = nn.Linear(embed_dim * 2, out_size)
        # Skip connection: E → out_size
        self.skip = nn.Linear(embed_dim, out_size)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.pe(self.embed(x))                       # (B, T, E)

        T_trim      = (T // self.patch_size) * self.patch_size
        x           = x[:, :T_trim, :]
        num_patches = T_trim // self.patch_size

        # Local: 2 stacked sparse blocks within each patch
        p = x.view(B * num_patches, self.patch_size, -1)
        p = self.local2(self.local1(p))
        p = p[:, -1, :].view(B, num_patches, -1)         # last-token aggregation

        # Global: 2 stacked sparse blocks across patch tokens
        g = self.global2(self.global1(p))                # (B, P, E)

        # 2-layer GRU → mean pool
        rnn_out, _ = self.rnn(g)
        ctx        = rnn_out.mean(dim=1)                  # (B, E)

        # Deep residual head
        h  = self.norm(ctx)
        h2 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h2)))
        out = self.fc3(h2) + self.skip(h)                # residual skip

        return out.view(B, self.pred_len, self.target_dim)  # (B, pred_len, TARGET_DIM)


# ── 8. Combined loss (0.5×MAE + 0.5×MSE — same as power consumption) ─────────

class CombinedLoss(nn.Module):
    """0.5 × MAE + 0.5 × MSE — balances spike-robustness and scale sensitivity."""
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)


# ── 9. Warmup-cosine LR scheduler (same as power consumption) ─────────────────

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

class WarmupCosineScheduler:
    """Linear warm-up for `warmup_epochs`, then cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.opt   = optimizer
        self.wu    = warmup_epochs
        self.total = total_epochs
        self.base  = base_lr
        self.min   = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.wu:
            lr = self.base * e / self.wu
        else:
            progress = (e - self.wu) / (self.total - self.wu)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr


# ── 10. Hyperparameters (same as power consumption) ───────────────────────────

EMBED_DIM  = 128
PATCH_SIZE = 12
TOPK       = 2
MAX_EPOCHS = 100
WARMUP     = 5
PATIENCE   = 10
BASE_LR    = 5e-4

results = {}

# ── 11. Training loop ─────────────────────────────────────────────────────────

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*60}\nPred len: {PRED_LEN}\n{'='*60}")
    set_seed(42)

    if len(X_scaled) < SEQ_LEN + PRED_LEN + 10:
        print(f"Skipping: insufficient data"); continue

    X_tr, y_tr = create_sequences(X_train_full, y_train_full, SEQ_LEN, PRED_LEN)
    X_va, y_va = create_sequences(X_val_full,   y_val_full,   SEQ_LEN, PRED_LEN)
    X_te, y_te = create_sequences(X_test_full,  y_test_full,  SEQ_LEN, PRED_LEN)

    if any(len(s) == 0 for s in [X_tr, X_va, X_te]):
        print(f"Skipping PRED_LEN={PRED_LEN}: no valid sequences"); continue

    print(f"Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")

    def make_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32)),
            batch_size=32, shuffle=shuffle)

    tr_ld = make_loader(X_tr, y_tr, shuffle=True)
    va_ld = make_loader(X_va, y_va)
    te_ld = make_loader(X_te, y_te)

    model = I3InformerV2Electricity(
        input_dim  = INPUT_DIM,
        target_dim = TARGET_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP, MAX_EPOCHS, BASE_LR)
    criterion = CombinedLoss(alpha=0.5)

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train(); tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        # ── validate ───────────────────────────────────────────────────────────
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_ld)

        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train {tr_loss:.6f} "
                  f"| val {va_loss:.6f} | lr {get_lr(optimizer):.6f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    # ── evaluate ───────────────────────────────────────────────────────────────
    model.load_state_dict(best_state); model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            y_pred_list.append(model(xb.to(device)).cpu().numpy())
            y_true_list.append(yb.numpy())

    y_true = np.concatenate(y_true_list, axis=0)   # (N, pred_len, TARGET_DIM)
    y_pred = np.concatenate(y_pred_list, axis=0)

    N, H, F = y_true.shape
    y_true_inv = inverse_y(y_true.reshape(-1, F))
    y_pred_inv = inverse_y(y_pred.reshape(-1, F))

    mae  = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2   = r2_score(y_true_inv, y_pred_inv)

    results[PRED_LEN] = dict(MAE=mae, RMSE=rmse, R2=r2,
                              state=best_state, X_te=X_te, y_te=y_te)
    print(f"  PRED_LEN={PRED_LEN} → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")


# ── 12. Summary ───────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("I3INFORMER V2 — ELECTRICITY LOAD FORECASTING (TEST SET)")
print(f"{'='*70}")
print(f"{'Pred len':<12}{'MAE':<18}{'RMSE':<18}{'R²'}")
print("-" * 60)
for pl in PRED_LENS:
    r = results.get(pl)
    if r and r['MAE'] is not None:
        print(f"{pl:<12}{r['MAE']:<18.4f}{r['RMSE']:<18.4f}{r['R2']:.4f}")
    else:
        print(f"{pl:<12}{'SKIPPED':<18}{'SKIPPED':<18}{'SKIPPED'}")


# ── 13. Plots — first 100 test-set steps, first MT_ column ───────────────────

valid_lengths = [pl for pl in PRED_LENS if results.get(pl, {}).get('MAE') is not None]

for PRED_LEN in valid_lengths:
    r = results[PRED_LEN]

    model_plot = I3InformerV2Electricity(
        input_dim  = INPUT_DIM,
        target_dim = TARGET_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)
    model_plot.load_state_dict(r['state'])
    model_plot.eval()

    y_true_list, y_pred_list = [], []
    plot_loader = DataLoader(
        TensorDataset(torch.tensor(r['X_te'], dtype=torch.float32),
                      torch.tensor(r['y_te'], dtype=torch.float32)),
        batch_size=32, shuffle=False)

    with torch.no_grad():
        for xb, yb in plot_loader:
            y_pred_list.append(model_plot(xb.to(device)).cpu().numpy())
            y_true_list.append(yb.numpy())

    y_true = np.concatenate(y_true_list, axis=0).reshape(-1, TARGET_DIM)
    y_pred = np.concatenate(y_pred_list, axis=0).reshape(-1, TARGET_DIM)

    y_true_inv = inverse_y(y_true)
    y_pred_inv = inverse_y(y_pred)

    plot_len = min(100, len(y_true_inv))

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv[:plot_len, 0], label='True',      linewidth=2)
    plt.plot(y_pred_inv[:plot_len, 0], label='Predicted', linewidth=2, alpha=0.8)
    plt.title(f"Electricity Load — I3InformerV2 | Pred Len {PRED_LEN} | "
              f"Feature: {feature_cols[0]}", fontsize=13)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Load', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f"i3informer_v2_electricity_predlen_{PRED_LEN}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: '{filename}'")

print("\nI3InformerV2 Electricity Load (full port) — completed!")


# In[]:
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


# In[]:
# I3Informer v2 — ILINet Forecasting
import warnings
warnings.filterwarnings('ignore')
import random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed=10):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & aggregate ILINet data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading ILINet data...")
df = pd.read_csv("ILINet.csv", header=None)
df.columns = [
    "REGION TYPE", "REGION", "YEAR", "WEEK", "% WEIGHTED ILI",
    "Original_%UNWEIGHTED ILI", "ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"
]
df["ILITOTAL"]       = pd.to_numeric(df["ILITOTAL"],        errors="coerce")
df["TOTAL PATIENTS"] = pd.to_numeric(df["TOTAL PATIENTS"],  errors="coerce")
df["%UNWEIGHTED ILI"] = (df["ILITOTAL"] / df["TOTAL PATIENTS"]) * 100
df = df.dropna(subset=["%UNWEIGHTED ILI"])

df_agg = df.groupby(['YEAR', 'WEEK']).agg(
    ILITOTAL=('ILITOTAL', 'sum'),
    TOTAL_PATIENTS=('TOTAL PATIENTS', 'sum')
).reset_index()
df_agg['%UNWEIGHTED ILI'] = (df_agg['ILITOTAL'] / df_agg['TOTAL_PATIENTS']) * 100
df_agg = df_agg.sort_values(['YEAR', 'WEEK']).reset_index(drop=True)
print(f" Loaded {len(df_agg)} weekly ILI records")

target_col = '%UNWEIGHTED ILI'

# Ensure YEAR and WEEK are numeric (they come out of groupby as object if the
# original CSV had mixed-type columns or a string header row)
df_agg['YEAR'] = pd.to_numeric(df_agg['YEAR'], errors='coerce')
df_agg['WEEK'] = pd.to_numeric(df_agg['WEEK'], errors='coerce')
df_agg.dropna(subset=['YEAR', 'WEEK'], inplace=True)
df_agg.reset_index(drop=True, inplace=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════

# D2 — Cyclical week-of-year features (ILI is strongly seasonal by week)
week_num = df_agg['WEEK'].astype(float)
df_agg['week_sin'] = np.sin(2 * np.pi * week_num / 52).astype(np.float32)
df_agg['week_cos'] = np.cos(2 * np.pi * week_num / 52).astype(np.float32)
TIME_FEATURES = ['week_sin', 'week_cos']
print("Cyclical week features added")

# D3 — Lag features: same week 1 year ago (52 weeks) and 2 years ago (104 weeks)
LAG_STEPS = [52, 104]
for lag in LAG_STEPS:
    df_agg[f'ili_lag{lag}'] = df_agg[target_col].shift(lag)
lag_cols = [f'ili_lag{lag}' for lag in LAG_STEPS]

df_agg.dropna(inplace=True)
df_agg.reset_index(drop=True, inplace=True)

ALL_FEATURES = [target_col] + TIME_FEATURES + lag_cols
print(f"Using {len(ALL_FEATURES)} features: {ALL_FEATURES}")

X_raw = df_agg[ALL_FEATURES].values.astype(np.float32)
y_raw = df_agg[[target_col]].values.astype(np.float32)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Scaling  (D1 — StandardScaler)
# ═══════════════════════════════════════════════════════════════════════════════
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
print("StandardScaler applied")

def inverse_y(arr):
    return y_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Sequence creation
# ═══════════════════════════════════════════════════════════════════════════════
def create_sequences(X, y, seq_len=48, pred_len=96):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN   = 48
PRED_LENS = [96, 120, 336, 720]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Model — I3Informer v2
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2):
        super().__init__()
        self.k, self.h = k, num_heads
        self.dh = embed_dim // num_heads
        self.qp = nn.Linear(embed_dim, embed_dim)
        self.kp = nn.Linear(embed_dim, embed_dim)
        self.vp = nn.Linear(embed_dim, embed_dim)
        self.op = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh   = self.h, self.dh
        q = self.qp(Q).view(B, L, H, Dh).transpose(1, 2)
        k = self.kp(K).view(B, L, H, Dh).transpose(1, 2)
        v = self.vp(V).view(B, L, H, Dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        ek = min(self.k, L)
        tv, ti = torch.topk(sc, ek, dim=-1)
        m = torch.full_like(sc, float('-inf'))
        m.scatter_(-1, ti, tv)
        a = torch.softmax(m, dim=-1)
        o = torch.matmul(a, v)
        return self.op(o.transpose(1, 2).reshape(B, L, E))


class SparseBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2, dropout=0.15):
        super().__init__()
        self.n1   = nn.LayerNorm(embed_dim)
        self.n2   = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        n = self.n1(x)
        x = x + self.drop(self.attn(n, n, n))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class I3InformerV2(nn.Module):
    def __init__(self, input_dim, embed_dim=128, patch_size=12,
                 pred_len=96, topk=2, seq_len=48, dropout=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.embed      = nn.Linear(input_dim, embed_dim)
        self.pe         = SinusoidalPE(embed_dim, max_len=seq_len+16, dropout=dropout)
        self.local1     = SparseBlock(embed_dim, 4, topk, dropout)
        self.local2     = SparseBlock(embed_dim, 4, topk, dropout)
        self.global1    = SparseBlock(embed_dim, 4, topk, dropout)
        self.global2    = SparseBlock(embed_dim, 4, topk, dropout)
        self.rnn        = nn.GRU(embed_dim, embed_dim, num_layers=2,
                                 batch_first=True, dropout=dropout)
        self.norm  = nn.LayerNorm(embed_dim)
        self.fc1   = nn.Linear(embed_dim, embed_dim * 4)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc3   = nn.Linear(embed_dim * 2, pred_len)
        self.skip  = nn.Linear(embed_dim, pred_len)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.pe(self.embed(x))
        T_trim      = (T // self.patch_size) * self.patch_size
        x           = x[:, :T_trim, :]
        num_patches = T_trim // self.patch_size
        p = x.view(B * num_patches, self.patch_size, -1)
        p = self.local2(self.local1(p))
        p = p[:, -1, :].view(B, num_patches, -1)
        g = self.global2(self.global1(p))
        rnn_out, _ = self.rnn(g)
        ctx = rnn_out.mean(dim=1)
        h  = self.norm(ctx)
        h2 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h2)))
        return self.fc3(h2) + self.skip(h)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Combined loss
# ═══════════════════════════════════════════════════════════════════════════════
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. LR scheduler
# ═══════════════════════════════════════════════════════════════════════════════
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.opt   = optimizer
        self.wu    = warmup_epochs
        self.total = total_epochs
        self.base  = base_lr
        self.min   = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.wu:
            lr = self.base * e / self.wu
        else:
            progress = (e - self.wu) / (self.total - self.wu)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

def get_lr(opt):
    return opt.param_groups[0]['lr']

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Training loop
# ═══════════════════════════════════════════════════════════════════════════════
EMBED_DIM  = 128
PATCH_SIZE = 12
TOPK       = 2
MAX_EPOCHS = 100
WARMUP     = 5
PATIENCE   = 10
BASE_LR    = 5e-4

results     = {}
best_states = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*70}\nPred len: {PRED_LEN} weeks\n{'='*70}")
    set_seed(10)

    min_required = SEQ_LEN + PRED_LEN + 100
    if len(X_scaled) < min_required:
        print(f" Skipping: insufficient data ({len(X_scaled)} < {min_required})"); continue

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) < 50:
        print(f"Skipping: too few sequences ({len(X_seq)})"); continue
    print(f"Created {len(X_seq)} sequences")

    n          = len(X_seq)
    tr         = int(0.70 * n)
    va         = int(0.85 * n)
    X_tr, y_tr = X_seq[:tr],      y_seq[:tr]
    X_va, y_va = X_seq[tr:va],    y_seq[tr:va]
    X_te, y_te = X_seq[va:],      y_seq[va:]
    print(f"Train={len(X_tr)} ({100*len(X_tr)/n:.1f}%)  "
          f"Val={len(X_va)} ({100*len(X_va)/n:.1f}%)  "
          f"Test={len(X_te)} ({100*len(X_te)/n:.1f}%)")

    def mk(X, y, shuf=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32).squeeze(-1)),
            batch_size=32, shuffle=shuf)

    tr_ld = mk(X_tr, y_tr, shuf=True)
    va_ld = mk(X_va, y_va)
    te_ld = mk(X_te, y_te)

    model = I3InformerV2(
        input_dim  = len(ALL_FEATURES),
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP, MAX_EPOCHS, BASE_LR)
    criterion = CombinedLoss(alpha=0.5)

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    print("Training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_ld)

        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train {tr_loss:.5f} | "
                  f"val {va_loss:.5f} | lr {get_lr(optimizer):.6f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    best_states[PRED_LEN] = best_state

    model.load_state_dict(best_state); model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            y_pred.append(model(xb.to(device)).cpu())
            y_true.append(yb)

    yt = inverse_y(torch.cat(y_true).numpy())
    yp = inverse_y(torch.cat(y_pred).numpy())

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)

    results[PRED_LEN] = dict(MAE=mae, RMSE=rmse, R2=r2,
                              X_te=X_te, y_te=y_te)
    print(f"  PRED_LEN={PRED_LEN} → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Summary
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("I3INFORMER V2 — ILINET FORECASTING — TEST SET RESULTS")
print(f"{'='*80}")
print(f"{'Pred len':<12}{'MAE':<16}{'RMSE':<16}{'R²'}")
print("-" * 55)
for pl in PRED_LENS:
    r = results.get(pl)
    if r:
        print(f"{pl:<12}{r['MAE']:<16.4f}{r['RMSE']:<16.4f}{r['R2']:.4f}")
    else:
        print(f"{pl:<12}{'SKIPPED':<16}{'SKIPPED':<16}{'SKIPPED'}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Visualisation — forecast plots
# ═══════════════════════════════════════════════════════════════════════════════
valid = [pl for pl in PRED_LENS if pl in results]
if valid:
    fig, axes = plt.subplots(len(valid), 1, figsize=(16, 5 * len(valid)))
    if len(valid) == 1:
        axes = [axes]

    for idx, PRED_LEN in enumerate(valid):
        r = results[PRED_LEN]
        X_te, y_te = r['X_te'], r['y_te']

        model.load_state_dict(best_states[PRED_LEN]); model.eval()

        last_X = X_te[-1:]
        last_y = y_te[-1:]

        with torch.no_grad():
            yp_sc = model(torch.tensor(last_X, dtype=torch.float32).to(device)).cpu().numpy()

        yt_inv      = inverse_y(last_y.reshape(-1, 1))
        yp_inv      = inverse_y(yp_sc.reshape(-1, 1))
        ctx_inv     = inverse_y(last_X.reshape(-1, len(ALL_FEATURES))[-24:, 0:1])

        ax           = axes[idx]
        t_ctx        = np.arange(SEQ_LEN - 24, SEQ_LEN)
        t_pred       = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

        ax.plot(t_ctx,  ctx_inv,  color='lightgray', linewidth=2,
                alpha=0.7, label='Context (last 24 weeks)')
        ax.plot(t_pred, yt_inv,   color='#2E86AB',   linewidth=2.5,
                alpha=0.9, label='Actual ILI %')
        ax.plot(t_pred, yp_inv,   color='#A23B72',   linewidth=2.5,
                linestyle='--', alpha=0.9, label='Predicted ILI %')
        ax.axvline(x=SEQ_LEN, color='black', linestyle=':', linewidth=1.5,
                   alpha=0.7, label='Prediction start')
        ax.set_title(
            f'ILINet Forecast — Input: {SEQ_LEN}w, Horizon: {PRED_LEN}w  '
            f'(MAE: {r["MAE"]:.4f}, RMSE: {r["RMSE"]:.4f}, R²: {r["R2"]:.4f})',
            fontsize=13)
        ax.set_xlabel('Week index'); ax.set_ylabel('Unweighted ILI (%)')
        ax.legend(fontsize=10); ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("ilinet_v2_all_predictions.png", dpi=150, bbox_inches='tight')
    print("\n Saved: ilinet_v2_all_predictions.png")

    # ── Comparison bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x     = np.arange(len(valid))
    width = 0.35
    maes  = [results[pl]['MAE']  for pl in valid]
    rmses = [results[pl]['RMSE'] for pl in valid]

    b1 = ax.bar(x - width/2, maes,  width, label='MAE',  color='#2E86AB',
                alpha=0.85, edgecolor='black')
    b2 = ax.bar(x + width/2, rmses, width, label='RMSE', color='#A23B72',
                alpha=0.85, edgecolor='black')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(valid)
    ax.set_xlabel('Prediction length (weeks)'); ax.set_ylabel('Error')
    ax.set_title('I3Informer V2 — ILINet performance across prediction lengths',
                 fontsize=13)
    ax.legend(); ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig("ilinet_v2_comparison.png", dpi=150, bbox_inches='tight')
    print("Saved: ilinet_v2_comparison.png")

    # ── Error distribution plot ─────────────────────────────────────────────
    if valid:
        PRED_LEN = valid[0]
        r        = results[PRED_LEN]
        model.load_state_dict(best_states[PRED_LEN]); model.eval()
        yt_all, yp_all = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(
                TensorDataset(torch.tensor(r['X_te'], dtype=torch.float32),
                              torch.tensor(r['y_te'], dtype=torch.float32).squeeze(-1)),
                    batch_size=32):
                yp_all.append(model(xb.to(device)).cpu())
                yt_all.append(yb)
        yt_inv = inverse_y(torch.cat(yt_all).numpy())
        yp_inv = inverse_y(torch.cat(yp_all).numpy())
        n_plot = min(100, len(yt_inv))

        fig, axes2 = plt.subplots(2, 1, figsize=(12, 8))
        axes2[0].plot(yt_inv[:n_plot], label='True',      linewidth=2, color='#2E86AB')
        axes2[0].plot(yp_inv[:n_plot], label='Predicted', linewidth=2,
                      color='#A23B72', linestyle='--', alpha=0.85)
        axes2[0].set_title(f"ILINet Forecast (I3Informer V2) — Pred len {PRED_LEN}",
                           fontsize=13)
        axes2[0].set_ylabel("Unweighted ILI (%)"); axes2[0].legend()
        axes2[0].grid(True, linestyle='--', alpha=0.5)

        err = yt_inv[:n_plot] - yp_inv[:n_plot]
        axes2[1].bar(range(n_plot), err, alpha=0.6, color='steelblue', label='Error')
        axes2[1].axhline(0, color='red', linewidth=1)
        axes2[1].set_title("Prediction error (true − predicted)", fontsize=12)
        axes2[1].set_ylabel("Error"); axes2[1].legend()
        axes2[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"ilinet_v2_error_predlen_{PRED_LEN}.png", dpi=120)
        print(f"Saved: ilinet_v2_error_predlen_{PRED_LEN}.png")

print("\n I3Informer v2 — ILINet forecasting completed!")


# In[]:
# I3Informer v2 — ETTh1 Multivariate Forecasting
import warnings
warnings.filterwarnings('ignore')

import random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 1. Load Dataset ───────────────────────────────────────────────────────────
df = pd.read_csv("ETTh1.csv")
df.columns = df.columns.str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

TARGET_COLS  = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
OUTPUT_DIM   = len(TARGET_COLS)   # 7 — predict all features
print(f"Target columns ({OUTPUT_DIM}): {TARGET_COLS}")
print(f"Rows after date cleaning: {len(df)}")


# ── 2. Cyclical Time Features ─────────────────────────────────────────────────
df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour        / 24).astype(np.float32)
df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour        / 24).astype(np.float32)
df['dow_sin']  = np.sin(2 * np.pi * df['date'].dt.dayofweek   /  7).astype(np.float32)
df['dow_cos']  = np.cos(2 * np.pi * df['date'].dt.dayofweek   /  7).astype(np.float32)
TIME_FEATURES  = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
print("Cyclical time features added:", TIME_FEATURES)


# ── 3. Lag Features (t-24, t-48 for every target column) ─────────────────────
LAG_STEPS = [24, 48]
lag_cols   = []
for lag in LAG_STEPS:
    for col in TARGET_COLS:
        new_col = f"{col}_lag{lag}"
        df[new_col] = df[col].shift(lag)
        lag_cols.append(new_col)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Lag features added ({len(lag_cols)}): {lag_cols[:4]} ...")
print(f"Rows after dropping NaN from lags: {len(df)}")


# ── 4. Build Feature Matrix & Target Matrix ───────────────────────────────────
# Input features  : 7 targets + 4 time + 14 lag cols = 25 features
# Output targets  : 7 columns (all ETTh1 variables, multivariate)
ALL_FEATURES = TARGET_COLS + TIME_FEATURES + lag_cols
INPUT_DIM    = len(ALL_FEATURES)
print(f"\nTotal input features: {INPUT_DIM}")
print(f"  Targets    : {len(TARGET_COLS)}")
print(f"  Time feats : {len(TIME_FEATURES)}")
print(f"  Lag feats  : {len(lag_cols)}")

X_raw = df[ALL_FEATURES].values.astype(np.float32)
y_raw = df[TARGET_COLS].values.astype(np.float32)


# ── 5. Scaling ────────────────────────────────────────────────────────────────
# Separate scalers for input and output (same as Temperature script pattern)
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
print("StandardScaler applied to inputs and targets.")

def inverse_y(arr):
    """Inverse-transform predictions/true values back to original scale."""
    arr = arr.reshape(-1, OUTPUT_DIM)
    return y_scaler.inverse_transform(arr)


# ── 6. Sequence Creation ──────────────────────────────────────────────────────
def create_sequences(X, y, seq_len, pred_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + pred_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN   = 48
PRED_LENS = [96, 120, 336, 720]


# ── 7. Model Definition: I3InformerV2 (identical to Temperature version) ─────

class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2):
        super().__init__()
        self.k, self.h = k, num_heads
        self.dh = embed_dim // num_heads
        self.qp = nn.Linear(embed_dim, embed_dim)
        self.kp = nn.Linear(embed_dim, embed_dim)
        self.vp = nn.Linear(embed_dim, embed_dim)
        self.op = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh   = self.h, self.dh
        q = self.qp(Q).view(B, L, H, Dh).transpose(1, 2)
        k = self.kp(K).view(B, L, H, Dh).transpose(1, 2)
        v = self.vp(V).view(B, L, H, Dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        ek = min(self.k, L)
        tv, ti = torch.topk(sc, ek, dim=-1)
        m = torch.full_like(sc, float('-inf'))
        m.scatter_(-1, ti, tv)
        a = torch.softmax(m, dim=-1)
        a = torch.nan_to_num(a, nan=0.0)   # guard all-inf rows
        o = torch.matmul(a, v)
        return self.op(o.transpose(1, 2).reshape(B, L, E))


class SparseBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2, dropout=0.15):
        super().__init__()
        self.n1   = nn.LayerNorm(embed_dim)
        self.n2   = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        n = self.n1(x)
        x = x + self.drop(self.attn(n, n, n))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class I3InformerV2(nn.Module):
    """
    I3Informer V2 — Multivariate ETTh1 version.

    Identical to the Temperature model except the final projection outputs
    pred_len * output_dim values instead of just pred_len, reshaped to
    (B, pred_len, output_dim).

    Architecture:
      embed → sinusoidal PE
      → 2× local SparseBlock (within patches)
      → last-token per patch
      → 2× global SparseBlock (across patches)
      → 2-layer GRU → mean-pool → context vector
      → deep head (E→4E→2E→pred_len*output_dim)  +  skip (E→pred_len*output_dim)
      → reshape (B, pred_len, output_dim)
    """
    def __init__(self, input_dim, output_dim, embed_dim=128, patch_size=12,
                 pred_len=96, topk=2, seq_len=48, dropout=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.pred_len   = pred_len
        self.output_dim = output_dim

        self.embed   = nn.Linear(input_dim, embed_dim)
        self.pe      = SinusoidalPE(embed_dim, max_len=seq_len + 16, dropout=dropout)

        # 2 local + 2 global sparse blocks (same as Temperature)
        self.local1  = SparseBlock(embed_dim, 4, topk, dropout)
        self.local2  = SparseBlock(embed_dim, 4, topk, dropout)
        self.global1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.global2 = SparseBlock(embed_dim, 4, topk, dropout)

        self.rnn  = nn.GRU(embed_dim, embed_dim, num_layers=2,
                           batch_first=True, dropout=dropout)

        # Deep projection head (mirrors Temperature)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1  = nn.Linear(embed_dim, embed_dim * 4)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc3  = nn.Linear(embed_dim * 2, pred_len * output_dim)

        # Skip connection
        self.skip = nn.Linear(embed_dim, pred_len * output_dim)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.pe(self.embed(x))                              # (B, T, E)

        T_trim      = (T // self.patch_size) * self.patch_size
        x           = x[:, :T_trim, :]
        num_patches = T_trim // self.patch_size

        # Local blocks within patches
        p = x.view(B * num_patches, self.patch_size, -1)
        p = self.local2(self.local1(p))
        p = p[:, -1, :].view(B, num_patches, -1)               # last token

        # Global blocks across patches
        g = self.global2(self.global1(p))                       # (B, P, E)

        # GRU → mean pool
        rnn_out, _ = self.rnn(g)
        ctx        = rnn_out.mean(dim=1)                        # (B, E)

        # Deep head + skip
        h  = self.norm(ctx)
        h2 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h2)))
        out = self.fc3(h2) + self.skip(h)                       # (B, pred_len*output_dim)

        return out.view(B, self.pred_len, self.output_dim)      # (B, pred_len, D)


# ── 8. Combined Loss ──────────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)


# ── 9. Warmup Cosine Scheduler ────────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.opt   = optimizer
        self.wu    = warmup_epochs
        self.total = total_epochs
        self.base  = base_lr
        self.min   = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.wu:
            lr = self.base * e / self.wu
        else:
            progress = (e - self.wu) / (self.total - self.wu)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

def get_lr(opt):
    return opt.param_groups[0]['lr']


# ── 10. Training Loop ─────────────────────────────────────────────────────────
EMBED_DIM  = 128
PATCH_SIZE = 12
TOPK       = 2
MAX_EPOCHS = 100
WARMUP     = 5
PATIENCE   = 10
BASE_LR    = 5e-4

results           = {}
best_model_states = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*70}")
    print(f"Processing Prediction Length: {PRED_LEN}")
    print(f"{'='*70}")
    set_seed(42)

    min_required = SEQ_LEN + PRED_LEN + 10
    if len(X_scaled) < min_required:
        print(f"Skipping PRED_LEN={PRED_LEN}: insufficient data")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) == 0:
        print(f"Skipping PRED_LEN={PRED_LEN}: no valid sequences")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue

    print(f"Created {len(X_seq)} sequences")

    n          = len(X_seq)
    tr         = int(0.70 * n)
    va         = int(0.15 * n)
    X_tr, y_tr = X_seq[:tr],      y_seq[:tr]
    X_va, y_va = X_seq[tr:tr+va], y_seq[tr:tr+va]
    X_te, y_te = X_seq[tr+va:],   y_seq[tr+va:]

    print(f"  Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    def make_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32)),
            batch_size=32, shuffle=shuffle)

    tr_ld = make_loader(X_tr, y_tr, shuffle=True)
    va_ld = make_loader(X_va, y_va)
    te_ld = make_loader(X_te, y_te)

    model = I3InformerV2(
        input_dim  = INPUT_DIM,
        output_dim = OUTPUT_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP, MAX_EPOCHS, BASE_LR)
    criterion = CombinedLoss(alpha=0.5)

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    print("Training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        # ── Validate ──
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_ld)

        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train {tr_loss:.6f} | "
                  f"val {va_loss:.6f} | lr {get_lr(optimizer):.6f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Evaluate on Test Set ──────────────────────────────────────────────────
    model.load_state_dict(best_state)
    best_model_states[PRED_LEN] = best_state
    print(f"  Best val loss: {best_val:.6f}")

    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            pred = model(xb.to(device)).cpu().numpy()
            y_true_list.append(yb.numpy())
            y_pred_list.append(pred)

    y_true_scaled = np.concatenate(y_true_list, axis=0).reshape(-1, OUTPUT_DIM)
    y_pred_scaled = np.concatenate(y_pred_list, axis=0).reshape(-1, OUTPUT_DIM)

    y_true_inv = inverse_y(y_true_scaled)
    y_pred_inv = inverse_y(y_pred_scaled)

    mae  = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2   = r2_score(y_true_inv, y_pred_inv)

    results[PRED_LEN] = {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'X_te': X_te, 'y_te': y_te,
    }
    print(f"  PRED_LEN={PRED_LEN} → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")


# ── 11. Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("I3INFORMER V2 — ETTh1 MULTIVARIATE FORECASTING (TEST SET)")
print(f"{'='*80}")
print(f"{'Pred Len':<12}{'MAE':<18}{'RMSE':<18}{'R²'}")
print("-" * 60)
for pl in PRED_LENS:
    r = results.get(pl, {})
    if r.get('MAE') is not None:
        print(f"{pl:<12}{r['MAE']:<18.4f}{r['RMSE']:<18.4f}{r['R2']:.4f}")
    else:
        print(f"{pl:<12}{'SKIPPED':<18}{'SKIPPED':<18}{'SKIPPED'}")
print(f"{'='*80}")


# ── 12. Plots ─────────────────────────────────────────────────────────────────
# Two-panel plot (forecast + error bar) for OT column — mirrors Temperature script
valid_lengths = [pl for pl in PRED_LENS if results.get(pl, {}).get('MAE') is not None]
ot_idx        = TARGET_COLS.index("OT")

for PRED_LEN in valid_lengths:
    set_seed(42)
    r = results[PRED_LEN]

    model_plot = I3InformerV2(
        input_dim  = INPUT_DIM,
        output_dim = OUTPUT_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)
    model_plot.load_state_dict(best_model_states[PRED_LEN])
    model_plot.eval()

    plot_loader = DataLoader(
        TensorDataset(torch.tensor(r['X_te'], dtype=torch.float32),
                      torch.tensor(r['y_te'], dtype=torch.float32)),
        batch_size=32, shuffle=False)

    yt_all, yp_all = [], []
    with torch.no_grad():
        for xb, yb in plot_loader:
            yp_all.extend(model_plot(xb.to(device)).cpu().numpy())
            yt_all.extend(yb.numpy())

    yt_inv = inverse_y(np.array(yt_all).reshape(-1, OUTPUT_DIM))[:, ot_idx]
    yp_inv = inverse_y(np.array(yp_all).reshape(-1, OUTPUT_DIM))[:, ot_idx]
    n_plot = min(100, len(yt_inv))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(yt_inv[:n_plot], label='True',      linewidth=2)
    axes[0].plot(yp_inv[:n_plot], label='Predicted', linewidth=2, alpha=0.8)
    axes[0].set_title(
        f"ETTh1 Forecast (I3Informer V2) — Pred Len {PRED_LEN} | Feature: OT",
        fontsize=13)
    axes[0].set_ylabel("OT (Oil Temperature)", fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    err = yt_inv[:n_plot] - yp_inv[:n_plot]
    axes[1].bar(range(n_plot), err, alpha=0.6, color='steelblue', label='Error')
    axes[1].axhline(0, color='red', linewidth=1)
    axes[1].set_title("Prediction Error (true − predicted)", fontsize=12)
    axes[1].set_xlabel("Time Step", fontsize=11)
    axes[1].set_ylabel("Error", fontsize=11)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fname = f"etth1_i3informer_v2_predlen_{PRED_LEN}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: '{fname}'")

print("\nI3Informer V2 — ETTh1 experiments completed!")


# In[]:
# I3Informer v2 — ETTm1
import warnings
warnings.filterwarnings('ignore')

import random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 1. Load Dataset ───────────────────────────────────────────────────────────
df = pd.read_csv("ETTm1.csv")          # ← rename to your actual filename
df.columns = df.columns.str.strip()

# Drop unnamed index column if present (e.g. the "1, 2, 3…" column)
unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
if unnamed:
    df.drop(columns=unnamed, inplace=True)
    print(f"Dropped unnamed columns: {unnamed}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

TARGET_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
OUTPUT_DIM  = len(TARGET_COLS)

# Verify columns exist
missing = [c for c in TARGET_COLS if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}. Available: {list(df.columns)}")

print(f"Target columns ({OUTPUT_DIM}): {TARGET_COLS}")
print(f"Rows after date cleaning: {len(df)}")

# Detect frequency
if len(df) > 1:
    delta = (df["date"].iloc[1] - df["date"].iloc[0]).total_seconds() / 60
    print(f"Detected frequency: {delta:.0f} minutes per step")
else:
    delta = 15
    print("Could not detect frequency — assuming 15 min")

assert abs(delta - 15) < 1, (
    f"Expected 15-min data but got {delta}-min steps. "
    "Adjust LAG_STEPS and SEQ_LEN accordingly.")


# ── 2. Cyclical Time Features ─────────────────────────────────────────────────
# For 15-min data we add minute-of-day (0-95) on top of hour and day-of-week
df['hour_sin']   = np.sin(2 * np.pi * df['date'].dt.hour         / 24 ).astype(np.float32)
df['hour_cos']   = np.cos(2 * np.pi * df['date'].dt.hour         / 24 ).astype(np.float32)
df['minute_sin'] = np.sin(2 * np.pi * df['date'].dt.minute       / 60 ).astype(np.float32)
df['minute_cos'] = np.cos(2 * np.pi * df['date'].dt.minute       / 60 ).astype(np.float32)
df['dow_sin']    = np.sin(2 * np.pi * df['date'].dt.dayofweek    /  7 ).astype(np.float32)
df['dow_cos']    = np.cos(2 * np.pi * df['date'].dt.dayofweek    /  7 ).astype(np.float32)
TIME_FEATURES    = ['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                    'dow_sin',  'dow_cos']
print(f"Cyclical time features added ({len(TIME_FEATURES)}): {TIME_FEATURES}")


# ── 3. Lag Features ───────────────────────────────────────────────────────────
# 15-min resolution: 96 steps = 24 h,  192 steps = 48 h
LAG_STEPS = [96, 192]
lag_cols  = []
for lag in LAG_STEPS:
    for col in TARGET_COLS:
        new_col       = f"{col}_lag{lag}"
        df[new_col]   = df[col].shift(lag)
        lag_cols.append(new_col)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Lag features added ({len(lag_cols)}): showing first 4 → {lag_cols[:4]} ...")
print(f"Rows after dropping NaN from lags: {len(df)}")


# ── 4. Feature Matrix & Target Matrix ────────────────────────────────────────
# 7 targets + 6 time + 14 lag = 27 input features
ALL_FEATURES = TARGET_COLS + TIME_FEATURES + lag_cols
INPUT_DIM    = len(ALL_FEATURES)
print(f"\nTotal input features : {INPUT_DIM}")
print(f"  Targets    : {len(TARGET_COLS)}")
print(f"  Time feats : {len(TIME_FEATURES)}")
print(f"  Lag feats  : {len(lag_cols)}")

X_raw = df[ALL_FEATURES].values.astype(np.float32)
y_raw = df[TARGET_COLS].values.astype(np.float32)


# ── 5. StandardScaler ────────────────────────────────────────────────────────
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X_raw).astype(np.float32)
y_scaled = y_scaler.fit_transform(y_raw).astype(np.float32)
print("StandardScaler applied to inputs and targets.")

def inverse_y(arr):
    return y_scaler.inverse_transform(arr.reshape(-1, OUTPUT_DIM))


# ── 6. Sequence Creation ──────────────────────────────────────────────────────
def create_sequences(X, y, seq_len, pred_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len : i + seq_len + pred_len])
    return np.array(Xs), np.array(ys)

# SEQ_LEN = 192  →  48 hours of 15-min context window
# PRED_LENS at 15-min resolution:
#   96  →  24 h    192 →  48 h    336 →  84 h    720 → 180 h
SEQ_LEN   = 192
PRED_LENS = [96, 192, 336, 720]


# ── 7. Model ──────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2):
        super().__init__()
        self.k, self.h = k, num_heads
        self.dh = embed_dim // num_heads
        self.qp = nn.Linear(embed_dim, embed_dim)
        self.kp = nn.Linear(embed_dim, embed_dim)
        self.vp = nn.Linear(embed_dim, embed_dim)
        self.op = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh   = self.h, self.dh
        q = self.qp(Q).view(B, L, H, Dh).transpose(1, 2)
        k = self.kp(K).view(B, L, H, Dh).transpose(1, 2)
        v = self.vp(V).view(B, L, H, Dh).transpose(1, 2)
        sc = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        ek = min(self.k, L)
        tv, ti = torch.topk(sc, ek, dim=-1)
        m = torch.full_like(sc, float('-inf'))
        m.scatter_(-1, ti, tv)
        a = torch.softmax(m, dim=-1)
        a = torch.nan_to_num(a, nan=0.0)
        o = torch.matmul(a, v)
        return self.op(o.transpose(1, 2).reshape(B, L, E))


class SparseBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=2, dropout=0.15):
        super().__init__()
        self.n1   = nn.LayerNorm(embed_dim)
        self.n2   = nn.LayerNorm(embed_dim)
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.drop = nn.Dropout(dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        n = self.n1(x)
        x = x + self.drop(self.attn(n, n, n))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class I3InformerV2(nn.Module):
    """
    I3Informer V2 — 15-min ETT multivariate.
    Identical architecture to the hourly ETTh1 version.
    patch_size=16 chosen so 192-step window gives 12 clean patches.
    """
    def __init__(self, input_dim, output_dim, embed_dim=128, patch_size=16,
                 pred_len=96, topk=2, seq_len=192, dropout=0.15):
        super().__init__()
        self.patch_size = patch_size
        self.pred_len   = pred_len
        self.output_dim = output_dim

        self.embed   = nn.Linear(input_dim, embed_dim)
        self.pe      = SinusoidalPE(embed_dim, max_len=seq_len + 32, dropout=dropout)

        self.local1  = SparseBlock(embed_dim, 4, topk, dropout)
        self.local2  = SparseBlock(embed_dim, 4, topk, dropout)
        self.global1 = SparseBlock(embed_dim, 4, topk, dropout)
        self.global2 = SparseBlock(embed_dim, 4, topk, dropout)

        self.rnn  = nn.GRU(embed_dim, embed_dim, num_layers=2,
                           batch_first=True, dropout=dropout)

        self.norm = nn.LayerNorm(embed_dim)
        self.fc1  = nn.Linear(embed_dim, embed_dim * 4)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.fc3  = nn.Linear(embed_dim * 2, pred_len * output_dim)
        self.skip = nn.Linear(embed_dim, pred_len * output_dim)

    def forward(self, x):
        B, T, _ = x.shape
        x = self.pe(self.embed(x))

        T_trim      = (T // self.patch_size) * self.patch_size
        x           = x[:, :T_trim, :]
        num_patches = T_trim // self.patch_size

        p = x.view(B * num_patches, self.patch_size, -1)
        p = self.local2(self.local1(p))
        p = p[:, -1, :].view(B, num_patches, -1)

        g = self.global2(self.global1(p))

        rnn_out, _ = self.rnn(g)
        ctx        = rnn_out.mean(dim=1)

        h  = self.norm(ctx)
        h2 = self.drop(self.act(self.fc1(h)))
        h2 = self.drop(self.act(self.fc2(h2)))
        out = self.fc3(h2) + self.skip(h)
        return out.view(B, self.pred_len, self.output_dim)


# ── 8. Loss & Scheduler ───────────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mae   = nn.L1Loss()
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mae(pred, target) + (1 - self.alpha) * self.mse(pred, target)


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-5):
        self.opt   = optimizer
        self.wu    = warmup_epochs
        self.total = total_epochs
        self.base  = base_lr
        self.min   = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        e = self.epoch
        if e <= self.wu:
            lr = self.base * e / self.wu
        else:
            progress = (e - self.wu) / (self.total - self.wu)
            lr = self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg['lr'] = lr

def get_lr(opt):
    return opt.param_groups[0]['lr']


# ── 9. Training Loop ──────────────────────────────────────────────────────────
EMBED_DIM  = 128
PATCH_SIZE = 16      # 192 / 16 = 12 patches  (clean division)
TOPK       = 2
MAX_EPOCHS = 100
WARMUP     = 5
PATIENCE   = 10
BASE_LR    = 5e-4

results           = {}
best_model_states = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*70}")
    print(f"Processing Prediction Length: {PRED_LEN}  ({PRED_LEN * 15 // 60}h {PRED_LEN * 15 % 60}min ahead)")
    print(f"{'='*70}")
    set_seed(42)

    if len(X_scaled) < SEQ_LEN + PRED_LEN + 10:
        print(f"Skipping PRED_LEN={PRED_LEN}: insufficient data")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) == 0:
        print(f"Skipping PRED_LEN={PRED_LEN}: no valid sequences")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue

    print(f"Created {len(X_seq)} sequences")

    n          = len(X_seq)
    tr         = int(0.70 * n)
    va         = int(0.15 * n)
    X_tr, y_tr = X_seq[:tr],      y_seq[:tr]
    X_va, y_va = X_seq[tr:tr+va], y_seq[tr:tr+va]
    X_te, y_te = X_seq[tr+va:],   y_seq[tr+va:]
    print(f"  Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    def make_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32),
                          torch.tensor(y, dtype=torch.float32)),
            batch_size=32, shuffle=shuffle)

    tr_ld = make_loader(X_tr, y_tr, shuffle=True)
    va_ld = make_loader(X_va, y_va)
    te_ld = make_loader(X_te, y_te)

    model = I3InformerV2(
        input_dim  = INPUT_DIM,
        output_dim = OUTPUT_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP, MAX_EPOCHS, BASE_LR)
    criterion = CombinedLoss(alpha=0.5)

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    print("Training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            loss   = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                va_loss += criterion(model(xb.to(device)), yb.to(device)).item()
        va_loss /= len(va_ld)

        scheduler.step()

        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train {tr_loss:.6f} | "
                  f"val {va_loss:.6f} | lr {get_lr(optimizer):.6f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    # ── Test Evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    best_model_states[PRED_LEN] = best_state
    print(f"  Best val loss: {best_val:.6f}")

    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            y_pred_list.append(model(xb.to(device)).cpu().numpy())
            y_true_list.append(yb.numpy())

    y_true_sc = np.concatenate(y_true_list, axis=0).reshape(-1, OUTPUT_DIM)
    y_pred_sc = np.concatenate(y_pred_list, axis=0).reshape(-1, OUTPUT_DIM)
    y_true_inv = inverse_y(y_true_sc)
    y_pred_inv = inverse_y(y_pred_sc)

    mae  = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2   = r2_score(y_true_inv, y_pred_inv)

    results[PRED_LEN] = {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'X_te': X_te, 'y_te': y_te,
    }
    print(f"  PRED_LEN={PRED_LEN} → MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")


# ── 10. Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("I3INFORMER V2 — ETT 15-MIN MULTIVARIATE FORECASTING (TEST SET)")
print(f"{'='*80}")
print(f"{'Pred Len':<12}{'Horizon':<12}{'MAE':<16}{'RMSE':<16}{'R²'}")
print("-" * 70)
for pl in PRED_LENS:
    r = results.get(pl, {})
    hrs = pl * 15 // 60
    mins = pl * 15 % 60
    horizon = f"{hrs}h{mins:02d}m"
    if r.get('MAE') is not None:
        print(f"{pl:<12}{horizon:<12}{r['MAE']:<16.4f}{r['RMSE']:<16.4f}{r['R2']:.4f}")
    else:
        print(f"{pl:<12}{horizon:<12}{'SKIPPED':<16}{'SKIPPED':<16}{'SKIPPED'}")
print(f"{'='*80}")


# ── 11. Plots ─────────────────────────────────────────────────────────────────
valid_lengths = [pl for pl in PRED_LENS if results.get(pl, {}).get('MAE') is not None]
ot_idx        = TARGET_COLS.index("OT")

for PRED_LEN in valid_lengths:
    set_seed(42)
    r = results[PRED_LEN]

    model_plot = I3InformerV2(
        input_dim  = INPUT_DIM,
        output_dim = OUTPUT_DIM,
        embed_dim  = EMBED_DIM,
        patch_size = PATCH_SIZE,
        pred_len   = PRED_LEN,
        topk       = TOPK,
        seq_len    = SEQ_LEN,
        dropout    = 0.15,
    ).to(device)
    model_plot.load_state_dict(best_model_states[PRED_LEN])
    model_plot.eval()

    plot_loader = DataLoader(
        TensorDataset(torch.tensor(r['X_te'], dtype=torch.float32),
                      torch.tensor(r['y_te'], dtype=torch.float32)),
        batch_size=32, shuffle=False)

    yt_all, yp_all = [], []
    with torch.no_grad():
        for xb, yb in plot_loader:
            yp_all.extend(model_plot(xb.to(device)).cpu().numpy())
            yt_all.extend(yb.numpy())

    yt_inv = inverse_y(np.array(yt_all).reshape(-1, OUTPUT_DIM))[:, ot_idx]
    yp_inv = inverse_y(np.array(yp_all).reshape(-1, OUTPUT_DIM))[:, ot_idx]
    n_plot = min(100, len(yt_inv))

    hrs     = PRED_LEN * 15 // 60
    mins_r  = PRED_LEN * 15 % 60
    horizon = f"{hrs}h {mins_r:02d}min"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(yt_inv[:n_plot], label='True',      linewidth=2)
    axes[0].plot(yp_inv[:n_plot], label='Predicted', linewidth=2, alpha=0.8)
    axes[0].set_title(
        f"ETT 15-Min Forecast (I3Informer V2) — Pred Len {PRED_LEN} ({horizon}) | OT",
        fontsize=13)
    axes[0].set_ylabel("OT (Oil Temperature)", fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    err = yt_inv[:n_plot] - yp_inv[:n_plot]
    axes[1].bar(range(n_plot), err, alpha=0.6, color='steelblue', label='Error')
    axes[1].axhline(0, color='red', linewidth=1)
    axes[1].set_title("Prediction Error (true − predicted)", fontsize=12)
    axes[1].set_xlabel("Time Step (15-min intervals)", fontsize=11)
    axes[1].set_ylabel("Error", fontsize=11)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fname = f"ettm1_i3informer_v2_predlen_{PRED_LEN}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: '{fname}'")

print("\nI3Informer V2 — ETT 15-Min experiments completed!")

