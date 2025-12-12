#Temperature Forecasting
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("../Data/temperatures.csv")
df = df[['Temperature']]
data = df.values


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_len=48, pred_len=96):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=5):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.head_dim = embed_dim // num_heads

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H = self.num_heads
        Dh = self.head_dim

        Q = self.q_proj(Q).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(K).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(V).view(B, L, H, Dh).transpose(1, 2)


        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Dh)

        k = min(self.k, L)
        topk_val, topk_idx = torch.topk(logits, k=k, dim=-1)

        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits.scatter_(-1, topk_idx, topk_val)
        attn = torch.softmax(masked_logits, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, E)
        return self.out(out)

class InformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=5):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x

class Twinformer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=32, patch_size=6, pred_len=96, topk=4):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.patch_size = patch_size
        self.pred_len = pred_len

        self.local = InformerBlock(embed_dim, k=topk)
        self.global_ = InformerBlock(embed_dim, k=topk)

        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, pred_len)

    def forward(self, x):
        B, T, D = x.shape

        x = self.embed(x)
        T = (T // self.patch_size) * self.patch_size
        x = x[:, :T, :]
        num_patches = T // self.patch_size
        x = x.view(B * num_patches, self.patch_size, -1)
        x = self.local(x)

        x = x.mean(dim=1).view(B, num_patches, -1)
        x = self.global_(x)

        _, h = self.rnn(x)

        return self.fc(h.squeeze(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

results = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*60}")
    print(f"Processing Prediction Length: {PRED_LEN}")
    print(f"{'='*60}")

    min_required = SEQ_LEN + PRED_LEN + 10
    if len(data) < min_required:
        print(f"")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)

    if len(X_seq) == 0:
        print(f"PRED_LEN={PRED_LEN}: no valid sequences created")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    print(f"Created {len(X_seq)} sequences")

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).squeeze(-1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Twinformer(
        input_dim=1,
        embed_dim=32,
        patch_size=6,
        pred_len=PRED_LEN,
        topk=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()


    print("Training...")
    for epoch in range(20):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20 - Loss: {total_loss/len(loader):.5f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(yb.numpy())

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

    results[PRED_LEN] = {'MAE': mae, 'RMSE': rmse}
    print(f" Results for PRED_LEN={PRED_LEN}:")
    print(f" MAE  = {mae:.4f}")
    print(f" RMSE = {rmse:.4f}")

print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}")
print(f"{'Pred Len':<10} {'MAE':<15} {'RMSE':<15}")
print(f"{'-'*70}")
for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        print(f"{pred_len:<10} {results[pred_len]['MAE']:<15.4f} {results[pred_len]['RMSE']:<15.4f}")
    else:
        print(f"{pred_len:<10} {'SKIPPED':<15} {'SKIPPED':<15}")

valid_lengths = [pl for pl in PRED_LENS if results[pl]['MAE'] is not None]
if valid_lengths:
    PRED_LEN = valid_lengths[0]
    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).squeeze(-1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = Twinformer(
        input_dim=1,
        embed_dim=32,
        patch_size=6,
        pred_len=PRED_LEN,
        topk=4
    ).to(device)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(yb.numpy())

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)

    plt.figure(figsize=(12, 5))
    plot_len = min(100, len(y_true_inv))
    plt.plot(y_true_inv[:plot_len], label='True', linewidth=2)
    plt.plot(y_pred_inv[:plot_len], label='Predicted', linewidth=2, alpha=0.8)
    plt.title(f"Temperature Forecasting – Pred Len {PRED_LEN}", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"temperature_forecast {PRED_LEN}.png")
    print(f"\nPlot saved as 'temperature_forecast{PRED_LEN}.png'")


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("powerconsumption.csv")
df.dropna(inplace=True)

print("\nAvailable columns:")
print(df.columns.tolist())

possible_features = [
    'Temperature', 'Humidity', 'WindSpeed',
    'GeneralDiffuseFlows', 'DiffuseFlows'
]

features = [col for col in possible_features if col in df.columns]

if len(features) == 0:
    raise ValueError("No feature columns found in CSV. Please check column names.")

print("\nUsing features:", features)

target_col = 'PowerConsumption_Zone1'
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in CSV.")

X_raw = df[features].values
y_raw = df[[target_col]].values

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

def create_sequences(X, y, seq_len=48, pred_len=96):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - pred_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(X_seq), np.array(y_seq)

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=5):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.head_dim = embed_dim // num_heads

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H = self.num_heads
        Dh = self.head_dim

        Q = self.q_proj(Q).view(B, L, H, Dh).transpose(1,2)
        K = self.k_proj(K).view(B, L, H, Dh).transpose(1,2)
        V = self.v_proj(V).view(B, L, H, Dh).transpose(1,2)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Dh)
        topk_vals, topk_idx = torch.topk(logits, k=self.k, dim=-1)

        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits.scatter_(-1, topk_idx, topk_vals)

        attn = torch.softmax(masked_logits, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1,2).reshape(B, L, E)
        return self.out(out)


class SparseInformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=5):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = x + self.ffn(x)
        return x


class I3Informer(nn.Module):
    def __init__(self, input_dim, embed_dim=32, patch_size=6, pred_len=96, topk=5):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.patch_size = patch_size
        self.local = SparseInformerBlock(embed_dim, k=topk)
        self.global_ = SparseInformerBlock(embed_dim, k=topk)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, pred_len)

    def forward(self, x):
        B, T, D = x.shape
        x = self.embed(x)
        T = (T // self.patch_size) * self.patch_size
        x = x[:, :T, :]
        num_patches = T // self.patch_size

        x = x.view(B * num_patches, self.patch_size, -1)
        x = self.local(x)

        x = x.mean(dim=1).view(B, num_patches, -1)

        x = self.global_(x)
        _, h = self.rnn(x)

        return self.fc(h.squeeze(0))
    SEQ_LEN = 48
    PRED_LENS = [96, 120, 336, 720]
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

for pred_len in PRED_LENS:
    print(f"\n{'='*60}")
    print(f"Training with SEQ_LEN={SEQ_LEN}, PRED_LEN={pred_len}")
    print(f"{'='*60}")
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, pred_len)
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).squeeze(-1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = I3Informer(input_dim=len(features), pred_len=pred_len).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
 
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.5f}")
    

    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(pred)
    
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    
    y_true_inv = y_scaler.inverse_transform(y_true)
    y_pred_inv = y_scaler.inverse_transform(y_pred)
    
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2 = r2_score(y_true_inv, y_pred_inv)
    
    results[pred_len] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"\nResults for PRED_LEN={pred_len}:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    model.eval()
    
    test_seq_len = SEQ_LEN
    total_needed = test_seq_len + pred_len
    if len(X_scaled) < total_needed:
        print(f"⚠️ Skipping plot for pred_len={pred_len}: not enough data.")
    else:

        start_idx = len(X_scaled) - total_needed
        X_plot = X_scaled[start_idx:start_idx + test_seq_len].reshape(1, test_seq_len, -1)
        y_plot_true = y_scaled[start_idx + test_seq_len:start_idx + total_needed]

        X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            y_plot_pred_scaled = model(X_plot_tensor).cpu().numpy().reshape(-1, 1)

    
        y_plot_true_inv = y_scaler.inverse_transform(y_plot_true)
        y_plot_pred_inv = y_scaler.inverse_transform(y_plot_pred_scaled)

    
        plt.figure(figsize=(14, 6))
        time_true = np.arange(test_seq_len, test_seq_len + pred_len)
        time_context = np.arange(test_seq_len)

        context_power = y_scaler.inverse_transform(y_scaled[start_idx:start_idx + test_seq_len])
        plt.plot(time_context[-24:], context_power[-24:], 'lightgray', label='Context (last 24)', linewidth=1)

        plt.plot(time_true, y_plot_true_inv, 'b-', label='Actual', linewidth=2)
        plt.plot(time_true, y_plot_pred_inv, 'r--', label='Forecast', linewidth=2)
        plt.axvline(x=test_seq_len, color='k', linestyle=':', alpha=0.7)
        plt.title(f'Power Consumption Forecast (Pred Len = {pred_len})')
        plt.xlabel('Time Step')
        plt.ylabel('Power Consumption (kW)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"forecast_predlen_{pred_len}.png", dpi=150)
        plt.close()
print(f"Forecast plot saved: forecast_predlen_{pred_len}.png")

print(f"\n{'='*60}")
print("SUMMARY RESULTS - All Prediction Lengths")
print(f"{'='*60}")
print(f"{'Pred Len':<12} {'MAE':<15} {'RMSE':<15} {'R²':<12}")
print(f"{'-'*60}")

for pred_len in PRED_LENS:
    mae = results[pred_len]['MAE']
    rmse = results[pred_len]['RMSE']
    r2 = results[pred_len]['R2']
    print(f"{pred_len:<12} {mae:<15.4f} {rmse:<15.4f} {r2:<12.4f}")


# WEATHER
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv("../Data/weather_utf8.csv")

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

data = df[feature_cols].values.astype(np.float32)
print(f"✅ Dataset loaded: {data.shape}")

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_len=48, pred_len=96):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=5):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh = self.num_heads, self.head_dim

        Q = self.q_proj(Q).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(K).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(V).view(B, L, H, Dh).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Dh)

        k = min(self.k, logits.size(-1))
        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)

        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits.scatter_(-1, topk_idx, topk_vals)

        attn = torch.softmax(masked_logits, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(out)

class InformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=5):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, k)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        x = self.norm2(x + self.ffn(x))
        return x

class Twinformer(nn.Module):
    def __init__(self, input_dim=21, embed_dim=32, patch_size=6, pred_len=96, topk=5):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.patch_size = patch_size

        self.embed = nn.Linear(input_dim, embed_dim)
        self.local = InformerBlock(embed_dim, k=topk)
        self.global_ = InformerBlock(embed_dim, k=topk)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, pred_len * input_dim)

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.input_dim, f"Input dim {D} != expected {self.input_dim}"
        x = self.embed(x)

        T_new = (T // self.patch_size) * self.patch_size
        x = x[:, :T_new, :]
        num_patches = T_new // self.patch_size

        x = x.view(B * num_patches, self.patch_size, self.embed_dim)

        x = self.local(x)
        x = x.mean(dim=1).view(B, num_patches, self.embed_dim)

        x = self.global_(x)

        _, h_n = self.rnn(x)

        out = self.fc(h_n.squeeze(0))
        return out.view(B, self.pred_len, self.input_dim)

results_summary = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*70}")
    print(f"Training for Prediction Length: {PRED_LEN}")
    print(f"Input Length: {SEQ_LEN} | Prediction Length: {PRED_LEN}")
    print(f"{'='*70}")

    if len(data) < SEQ_LEN + PRED_LEN + 10:
        print(f"PRED_LEN={PRED_LEN}: insufficient data")
        results_summary[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) == 0:
        print(f"PRED_LEN={PRED_LEN}: no valid sequences")
        results_summary[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    print(f"Created {len(X_seq)} sequences")

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Twinformer(
        input_dim=len(feature_cols),
        embed_dim=32,
        patch_size=6,
        pred_len=PRED_LEN,
        topk=8
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("\n Training started...\n")
    for epoch in range(20):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/20 → Avg Loss: {avg_loss:.6f}")

    print("\n Evaluating...\n")
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu()
            y_pred_list.append(pred)
            y_true_list.append(yb)

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    B, H, F = y_true.shape
    y_true_flat = y_true.reshape(-1, F)
    y_pred_flat = y_pred.reshape(-1, F)

    y_true_inv = scaler.inverse_transform(y_true_flat)
    y_pred_inv = scaler.inverse_transform(y_pred_flat)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

    results_summary[PRED_LEN] = {'MAE': mae, 'RMSE': rmse}

    print(f"{'='*60}")
    print(f" Results for PRED_LEN={PRED_LEN}:")
    print(f"{'='*60}")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")

print(f"\n\n{'='*70}")
print("Weather result summary")
print(f"{'='*70}")
print(f"{'Pred Len':<15} {'MAE':<20} {'RMSE':<20}")
print(f"{'-'*70}")
for pred_len in PRED_LENS:
    if results_summary[pred_len]['MAE'] is not None:
        mae = results_summary[pred_len]['MAE']
        rmse = results_summary[pred_len]['RMSE']
        print(f"{pred_len:<15} {mae:<20.4f} {rmse:<20.4f}")
    else:
        print(f"{pred_len:<15} {'SKIPPED':<20} {'SKIPPED':<20}")
print(f"{'='*70}")

#Electricity Load

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("../Data/Electricity_load.csv")

if "Unnamed: 0" in df.columns:
    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")

feature_cols = [c for c in df.columns if c.startswith("MT_")]
data = df[feature_cols].values.astype(np.float32)
num_features = len(feature_cols)

print("Loaded features:", num_features)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_len=48, pred_len=96):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=5):
        super().__init__()
        self.top_k = top_k
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh = self.num_heads, self.head_dim

        Q = self.q_proj(Q).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(K).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(V).view(B, L, H, Dh).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Dh)

        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)

        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, top_idx, top_vals)

        attn = torch.softmax(masked_logits, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).reshape(B, L, E)
        return self.out(out)

class InformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=5):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, top_k)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = x + self.ffn(x)
        return x

class Twinformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, patch_size=6,
                 pred_len=96, top_k=5):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Linear(input_dim, embed_dim)

        self.local = InformerBlock(embed_dim, top_k=top_k)
        self.global_ = InformerBlock(embed_dim, top_k=top_k)

        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, pred_len * input_dim)

    def forward(self, x):
        B, T, D = x.shape
        x = self.embed(x)

        T = (T // self.patch_size) * self.patch_size
        x = x[:, :T, :]

        num_patches = T // self.patch_size
        x = x.view(B * num_patches, self.patch_size, -1)

        x = self.local(x)
        x = x.mean(dim=1).view(B, num_patches, -1)

        x = self.global_(x)

        _, h = self.rnn(x)
        out = self.fc(h.squeeze(0))

        return out.view(B, -1, D)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

results = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*70}")
    print(f"PROCESSING: {PRED_LEN}")
    print(f"{'='*70}")

    min_required = SEQ_LEN + PRED_LEN + 10
    if len(data) < min_required:
        print(f"PRED_LEN={PRED_LEN}: insufficient data ({len(data)} < {min_required})")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    X, y = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)

    if len(X) == 0:
        print(f"PRED_LEN={PRED_LEN}: no valid sequences created")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    print(f"Created {len(X)} sequences")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Twinformer(input_dim=num_features, embed_dim=128, patch_size=6, pred_len=PRED_LEN, top_k=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Training...")
    for epoch in range(20):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/20: Loss = {total_loss/len(loader):.6f}")

    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_true_all.append(yb.numpy())
            y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    yt = y_true.reshape(-1, num_features)
    yp = y_pred.reshape(-1, num_features)

    yt_inv = scaler.inverse_transform(yt)
    yp_inv = scaler.inverse_transform(yp)

    mae = np.mean(np.abs(yt_inv - yp_inv))
    rmse = np.sqrt(np.mean((yt_inv - yp_inv)**2))

    results[PRED_LEN] = {'MAE': mae, 'RMSE': rmse}
    print(f"Results for PRED_LEN={PRED_LEN}:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")

for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        mae = results[pred_len]['MAE']
        rmse = results[pred_len]['RMSE']
        print(f"{pred_len:<20} {mae:<20.4f} {rmse:<20.4f}")
    else:
        print(f"{pred_len:<20} {'SKIPPED':<20} {'SKIPPED':<20}")

print(f"{'='*80}")

#Idea
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("../Data/IDEA.csv")

if not pd.api.types.is_numeric_dtype(df.iloc[1]["Open"]):
    df = df.drop(index=1).reset_index(drop=True)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)


df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df.dropna(subset=["Close"], inplace=True)

data = df[["Close"]].values.astype(np.float32)


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]


def create_sequences(data, seq_len, pred_len):
    """Create sequences"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

datasets = {}
for pred_len in PRED_LENS:
    min_required = SEQ_LEN + pred_len + 10
    if len(data_scaled) < min_required:
        datasets[pred_len] = None
        continue

    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, pred_len)
    if len(X_seq) == 0:
        datasets[pred_len] = None
        continue

    datasets[pred_len] = (X_seq, y_seq)
    print(f"PRED_LEN={pred_len}: Created {len(X_seq)} sequences")

class TopKSparseAttention(nn.Module):
    """Top-K Sparse Attention"""
    def __init__(self, embed_dim, num_heads=4, top_k=5):
        super().__init__()
        self.top_k = top_k
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H = self.num_heads
        Dh = self.head_dim

        Q = self.q_proj(Q).view(B, L, H, Dh).transpose(1, 2)
        K = self.k_proj(K).view(B, L, H, Dh).transpose(1, 2)
        V = self.v_proj(V).view(B, L, H, Dh).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Dh)

        top_k = min(self.top_k, logits.shape[-1])
        top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)


        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, top_idx, top_vals)


        attn = torch.softmax(masked_logits, dim=-1)


        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, E)

        return self.out_proj(out)
class InformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=5, dropout=0.1):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, top_k)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = self.norm1(x)

        x = x + self.ffn(x)
        x = self.norm2(x)

        return x

class Twinformer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, num_heads=4, top_k=5,
                 patch_size=6, num_layers=2, dropout=0.1, pred_len=96):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.pred_len = pred_len
        self.embed = nn.Linear(input_dim, embed_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layers = nn.ModuleList([
            InformerBlock(embed_dim, num_heads, top_k, dropout)
            for _ in range(num_layers)
        ])


        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True, dropout=dropout if num_layers > 1 else 0)


        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, pred_len * input_dim)
        )

    def forward(self, x):
        B, T, D = x.shape

        x = self.embed(x) + self.pos_enc


        for layer in self.layers:
            x = layer(x)

        _, h = self.gru(x)
        h = h.squeeze(0)


        out = self.pred_head(h)
        out = out.view(B, self.pred_len, D)

        return out

def train_model(model, loader, optimizer, criterion, num_epochs, device, scheduler=None):
    """Train model"""
    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.5f}")

    return losses


def evaluate_model(model, loader, device, scaler):
    """Evaluate the model and compute metrics"""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(pred)

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

    return mae, rmse, y_true_inv, y_pred_inv

results = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*70}")
    print(f"PROCESSING PREDICTION LENGTH: {PRED_LEN}")
    print(f"{'='*70}")


    if datasets[PRED_LEN] is None:
        print(f"PRED_LEN={PRED_LEN}: insufficient data or no sequences")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'model': None}
        continue

    X_seq, y_seq = datasets[PRED_LEN]
    print(f"Sequences: {len(X_seq)}")

    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Twinformer(
        input_dim=1,
        embed_dim=64,
        num_heads=4,
        top_k=5,
        patch_size=6,
        num_layers=2,
        dropout=0.1,
        pred_len=PRED_LEN
    ).to(device)

    print(f"Model initialized")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.MSELoss()

    print("Training...")
    losses = train_model(model, loader, optimizer, criterion, num_epochs=20, device=device, scheduler=scheduler)

    print("Evaluating...")
    mae, rmse, y_true_inv, y_pred_inv = evaluate_model(model, loader, device, scaler)

    results[PRED_LEN] = {
        'MAE': mae,
        'RMSE': rmse,
        'model': model,
        'losses': losses,
        'y_true': y_true_inv,
        'y_pred': y_pred_inv
    }

    print(f"PRED_LEN={PRED_LEN} Results:")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")

print(f"\n{'='*80}")
print("Twinformer IDEA Stock Close Price Forecasting")
print(f"{'='*80}")
print(f"{'Prediction Length':<20} {'MAE':<20} {'RMSE':<20}")
print(f"{'-'*80}")

for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        mae = results[pred_len]['MAE']
        rmse = results[pred_len]['RMSE']
        print(f"{pred_len:<20} {mae:<20.4f} {rmse:<20.4f}")
    else:
        print(f"{pred_len:<20} {'SKIPPED':<20} {'SKIPPED':<20}")

print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Twinformer: Training Loss and Performance', fontsize=16, fontweight='bold')

for idx, pred_len in enumerate(PRED_LENS):
    if results[pred_len]['MAE'] is not None:
        ax = axes[idx // 2, idx % 2]


for idx, pred_len in enumerate(PRED_LENS):
    if results[pred_len]['MAE'] is not None:
        ax = axes[idx // 2, idx % 2]

        y_true = results[pred_len]['y_true'][:100]
        y_pred = results[pred_len]['y_pred'][:100]

if all(results[pl]['MAE'] is not None for pl in PRED_LENS):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Twinformer Performance Metrics', fontsize=14, fontweight='bold')

    pred_lens = [pl for pl in PRED_LENS if results[pl]['MAE'] is not None]
    mae_values = [results[pl]['MAE'] for pl in pred_lens]
    rmse_values = [results[pl]['RMSE'] for pl in pred_lens]

for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        y_true = results[pred_len]['y_true'][:5]
        y_pred = results[pred_len]['y_pred'][:5]

        print(f"\n Prediction Length: {pred_len}")
        print(f"{'Sample':<10} {'Actual (₹)':<20} {'Predicted (₹)':<20} {'Absolute Error (₹)':<20}")
        print(f"{'-'*75}")

        for i in range(len(y_true)):
            true_val = y_true[i, 0]
            pred_val = y_pred[i, 0]
            error = abs(true_val - pred_val)
            print(f"{i+1:<10} {true_val:<20.4f} {pred_val:<20.4f} {error:<20.4f}")

for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        y_true = results[pred_len]['y_true']
        y_pred = results[pred_len]['y_pred']
        errors = np.abs(y_true - y_pred)

#ETTh1 dataset

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("../Data/ETTh1.csv")  

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")

feature_cols = ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]
data = df[feature_cols].values.astype(np.float32)

num_features = len(feature_cols)
print("Features:", num_features)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_len=48, pred_len=96):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=5):
        super().__init__()
        self.top_k = top_k
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh = self.num_heads, self.head_dim

        Q = self.q(Q).view(B, L, H, Dh).transpose(1,2)
        K = self.k(K).view(B, L, H, Dh).transpose(1,2)
        V = self.v(V).view(B, L, H, Dh).transpose(1,2)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Dh)
        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)

        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, top_idx, top_vals)

        attn = torch.softmax(masked_logits, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1,2).reshape(B, L, E)
        return self.out(out)

class InformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, top_k=5):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, top_k)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = x + self.ffn(x)
        return x

class Twinformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, patch_size=6,
                 pred_len=96, top_k=5):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Linear(input_dim, embed_dim)

        self.local = InformerBlock(embed_dim, top_k=top_k)
        self.global_ = InformerBlock(embed_dim, top_k=top_k)

        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, pred_len * input_dim)

    def forward(self, x):
        B, T, D = x.shape
        x = self.embed(x)


        T2 = (T // self.patch_size) * self.patch_size
        x = x[:, :T2, :]
        num_patches = T2 // self.patch_size

        x = x.view(B * num_patches, self.patch_size, -1)
        x = self.local(x)
        x = x.mean(dim=1).view(B, num_patches, -1)

        x = self.global_(x)

        _, h = self.rnn(x)
        out = self.fc(h.squeeze(0))

        return out.view(B, -1, D)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

results = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*70}")
    print(f"PROCESSING PREDICTION LENGTH: {PRED_LEN}")
    print(f"{'='*70}")

    min_required = SEQ_LEN + PRED_LEN + 10
    if len(data) < min_required:
        print(f"PRED_LEN={PRED_LEN}: insufficient data ({len(data)} < {min_required})")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    X, y = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)

    if len(X) == 0:
        print(f"PRED_LEN={PRED_LEN}: no valid sequences created")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    print(f"Created {len(X)} sequences")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Twinformer(input_dim=num_features, embed_dim=64, patch_size=6, pred_len=PRED_LEN, top_k=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    print("Training...")
    for epoch in range(20):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/20: Loss = {total_loss/len(loader):.6f}")

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()

            y_true.append(yb.numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    yt = y_true.reshape(-1, num_features)
    yp = y_pred.reshape(-1, num_features)

    yt_inv = scaler.inverse_transform(yt)
    yp_inv = scaler.inverse_transform(yp)

    mae = np.mean(np.abs(yt_inv - yp_inv))
    rmse = np.sqrt(np.mean((yt_inv - yp_inv)**2))

    results[PRED_LEN] = {'MAE': mae, 'RMSE': rmse}
    print(f"Results for PRED_LEN={PRED_LEN}:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")

print(f"\n{'='*80}")
print("ETTh1")
print(f"{'='*80}")
print(f"{'Prediction Length':<20} {'MAE':<20} {'RMSE':<20}")
print(f"{'-'*80}")

for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        mae = results[pred_len]['MAE']
        rmse = results[pred_len]['RMSE']
        print(f"{pred_len:<20} {mae:<20.4f} {rmse:<20.4f}")
    else:
        print(f"{pred_len:<20} {'SKIPPED':<20} {'SKIPPED':<20}")

#ETTm1 dataset

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("../Data/ETTm1.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df = df.sort_values("date")

feature_cols = ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]
data = df[feature_cols].values.astype(np.float32)

num_features = len(feature_cols)
print("Number of features =", num_features)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_len=48, pred_len=96):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=5):
        super().__init__()
        self.top_k = top_k
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        B, L, E = Q.shape
        H, Dh = self.num_heads, self.head_dim

        Q = self.q(Q).view(B, L, H, Dh).transpose(1,2)
        K = self.k(K).view(B, L, H, Dh).transpose(1,2)
        V = self.v(V).view(B, L, H, Dh).transpose(1,2)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Dh)

        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)

        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, top_idx, top_vals)

        attn = torch.softmax(masked_logits, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1,2).reshape(B, L, E)
        return self.out(out)

class InformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, top_k=5):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, top_k)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(x, x, x)
        x = x + self.ffn(x)
        return x

class Twinformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, patch_size=6,
                 pred_len=96, top_k=5):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Linear(input_dim, embed_dim)

        self.local = InformerBlock(embed_dim, top_k=top_k)
        self.global_ = InformerBlock(embed_dim, top_k=top_k)

        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, pred_len * input_dim)

    def forward(self, x):
        B, T, D = x.shape
        x = self.embed(x)

        T2 = (T // self.patch_size) * self.patch_size
        x = x[:, :T2, :]
        num_patches = T2 // self.patch_size

        x = x.view(B * num_patches, self.patch_size, -1)
        x = self.local(x)
        x = x.mean(dim=1).view(B, num_patches, -1)

        x = self.global_(x)
        _, h = self.rnn(x)

        out = self.fc(h.squeeze(0))
        return out.view(B, -1, D)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

results = {}

for PRED_LEN in PRED_LENS:
    print(f"\n{'='*70}")
    print(f"PROCESSING PREDICTION LENGTH: {PRED_LEN}")
    print(f"{'='*70}")

    min_required = SEQ_LEN + PRED_LEN + 10
    if len(data) < min_required:
        print(f"Skipping PRED_LEN={PRED_LEN}: insufficient data ({len(data)} < {min_required})")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    X, y = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)

    if len(X) == 0:
        print(f"PRED_LEN={PRED_LEN}: no valid sequences created")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    print(f"Created {len(X)} sequences")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Twinformer(input_dim=num_features, embed_dim=64, patch_size=6, pred_len=PRED_LEN, top_k=5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    print("Training...")
    for epoch in range(20):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1}/20: Loss = {total/len(loader):.6f}")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_true.append(yb.numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    yt = y_true.reshape(-1, num_features)
    yp = y_pred.reshape(-1, num_features)

    yt_inv = scaler.inverse_transform(yt)
    yp_inv = scaler.inverse_transform(yp)

    mae = np.mean(np.abs(yt_inv - yp_inv))
    rmse = np.sqrt(np.mean((yt_inv - yp_inv)**2))

    results[PRED_LEN] = {'MAE': mae, 'RMSE': rmse}
    print(f"Results for PRED_LEN={PRED_LEN}:")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")

print(f"\n{'='*80}")
print(f"{'='*80}")
print(f"{'Prediction Length':<20} {'MAE':<20} {'RMSE':<20}")
print(f"{'-'*80}")

for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        mae = results[pred_len]['MAE']
        rmse = results[pred_len]['RMSE']
        print(f"{pred_len:<20} {mae:<20.4f} {rmse:<20.4f}")
    else:
        print(f"{pred_len:<20} {'SKIPPED':<20} {'SKIPPED':<20}")

print(f"{'='*80}")