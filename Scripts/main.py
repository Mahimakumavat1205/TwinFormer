# Temperature Forecasting 

import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)


df = pd.read_csv("temperatures.csv")
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
    def __init__(self, embed_dim, num_heads=4, k=3):
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
    def __init__(self, embed_dim, num_heads=4, k=3):
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


class I3Informer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=32, patch_size=6, pred_len=96, topk=3):
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
print(f"Using device: {device}\n")

results = {}
best_models = {}

for PRED_LEN in PRED_LENS:
    print(f"Processing Prediction Length: {PRED_LEN}")
    
    
    set_seed(10)
    
    min_required = SEQ_LEN + PRED_LEN + 100
    if len(data) < min_required:
        print(f" Skipping: insufficient data\n")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue

    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    
    if len(X_seq) < 100:
        print(f" Skipping: too few sequences\n")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue
    
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))
    
    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    print(f" Dataset split:")
    print(f"   Train: {len(X_train)} ({len(X_train)/len(X_seq)*100:.1f}%)")
    print(f"   Val:   {len(X_val)} ({len(X_val)/len(X_seq)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} ({len(X_test)/len(X_seq)*100:.1f}%)")


    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).squeeze(-1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).squeeze(-1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).squeeze(-1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    
    model = I3Informer(
        input_dim=1, embed_dim=32, patch_size=6, 
        pred_len=PRED_LEN, topk=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

  
    print("\n Training...")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(20):
       
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/20 - Train: {train_loss:.5f}, Val: {val_loss:.5f}")
    
    
    model.load_state_dict(best_model_state)
    best_models[PRED_LEN] = best_model_state
    
    print(f"\n Testing on held-out test set...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
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
    r2 = r2_score(y_true_inv, y_pred_inv)
    
    results[PRED_LEN] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"\n Test Results:")
    print(f"   MAE  = {mae:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   R²   = {r2:.4f}")
    print()


print("FINAL RESULTS SUMMARY")
print(f"{'Pred Len':<12} {'MAE':<15} {'RMSE':<15} {'R² Score':<15}")
for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        print(f"{pred_len:<12} {results[pred_len]['MAE']:<15.4f} "
              f"{results[pred_len]['RMSE']:<15.4f} {results[pred_len]['R2']:<15.4f}")
    else:
        print(f"{pred_len:<12} {'SKIPPED':<15} {'SKIPPED':<15} {'SKIPPED':<15}")

valid_lengths = [pl for pl in PRED_LENS if results[pl]['MAE'] is not None]
if valid_lengths:
    PRED_LEN = valid_lengths[0]
    print(f"\n Creating plot for prediction length {PRED_LEN}...")
    
    set_seed(10)
    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))
    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]
    
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).squeeze(-1)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = I3Informer(
        input_dim=1, embed_dim=32, patch_size=6,
        pred_len=PRED_LEN, topk=3
    ).to(device)
    model.load_state_dict(best_models[PRED_LEN])
    
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(yb.numpy())
    
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    plt.figure(figsize=(14, 6))
    plot_len = min(100, len(y_true_inv))
    plt.plot(y_true_inv[:plot_len].flatten(), label='True', linewidth=2, alpha=0.8)
    plt.plot(y_pred_inv[:plot_len].flatten(), label='Predicted', linewidth=2, alpha=0.8)
    plt.title(f" ", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Temperature", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"Temperature_{PRED_LEN}.png", dpi=150)
    print(f"Temperature_{PRED_LEN}.png'")


#PowerConsumption
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

df = pd.read_csv("D:\current_folder\powerconsumption.csv")
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

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=3):
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
    def __init__(self, embed_dim, num_heads=4, k=3):
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
    def __init__(self, input_dim, embed_dim=32, patch_size=6, pred_len=96, topk=3):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Random seed: 10 (for reproducibility)")

results = {}

for PRED_LEN in PRED_LENS:
    
    print(f"Processing Prediction Length: {PRED_LEN}")
    
    
    set_seed(10)  

    min_required = SEQ_LEN + PRED_LEN + 10
    if len(X_scaled) < min_required:
        print(f" Skipping PRED_LEN={PRED_LEN}: insufficient data ({len(X_scaled)} < {min_required})")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_LEN)
    
    if len(X_seq) == 0:
        print(f" Skipping PRED_LEN={PRED_LEN}: no valid sequences created")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue
        
    print(f"Created {len(X_seq)} sequences")
    
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))
    test_size = len(X_seq) - train_size - val_size

    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

    print(f" Data Split:")
    print(f"   Training:   {len(X_train)} sequences ({100*len(X_train)/len(X_seq):.1f}%)")
    print(f"   Validation: {len(X_val)} sequences ({100*len(X_val)/len(X_seq):.1f}%)")
    print(f"   Testing:    {len(X_test)} sequences ({100*len(X_test)/len(X_seq):.1f}%)")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                           torch.tensor(y_train, dtype=torch.float32).squeeze(-1)),
                             batch_size=32, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                         torch.tensor(y_val, dtype=torch.float32).squeeze(-1)),
                           batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                          torch.tensor(y_test, dtype=torch.float32).squeeze(-1)),
                           batch_size=32, shuffle=False)

    
    model = I3Informer(
        input_dim=len(features),
        embed_dim=32,
        patch_size=6,
        pred_len=PRED_LEN,
        topk=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

  
    print("Training...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(20):
        
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20 - Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

   
    model.load_state_dict(best_model_state)
    print(f"Best validation loss: {best_val_loss:.5f}")

   
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(yb.numpy())

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_inv = y_scaler.inverse_transform(y_true)
    y_pred_inv = y_scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2 = r2_score(y_true_inv, y_pred_inv)

    results[PRED_LEN] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    print(f" Test Results for PRED_LEN={PRED_LEN}:")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    R²   = {r2:.4f}")

print("FINAL TEST RESULTS SUMMARY")
print(f"{'Pred Len':<10} {'MAE':<15} {'RMSE':<15} {'R²':<12}")
for pred_len in PRED_LENS:
    res = results[pred_len]
    if res['MAE'] is not None:
        print(f"{pred_len:<10} {res['MAE']:<15.4f} {res['RMSE']:<15.4f} {res['R2']:<12.4f}")
    else:
        print(f"{pred_len:<10} {'SKIPPED':<15} {'SKIPPED':<15} {'SKIPPED':<12}")

valid_lengths = [pl for pl in PRED_LENS if results[pl]['MAE'] is not None]
if valid_lengths:
    PRED_LEN = valid_lengths[0]
    set_seed(10)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_LEN)
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).squeeze(-1)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = I3Informer(
        input_dim=len(features),
        embed_dim=32,
        patch_size=6,
        pred_len=PRED_LEN,
        topk=3
    ).to(device)
    model.load_state_dict(best_model_state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(pred)

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_inv = y_scaler.inverse_transform(y_true)
    y_pred_inv = y_scaler.inverse_transform(y_pred)

    total_context = min(24, SEQ_LEN)
    plot_len = min(100, len(y_true_inv))
    
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv[:plot_len].flatten()[:100], label='True', linewidth=2)
    plt.plot(y_pred_inv[:plot_len].flatten()[:100], label='Predicted', linewidth=2, alpha=0.8)
    plt.title(f"Power Consumption Forecast- Test Set - Pred Len {PRED_LEN}", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f'power_forecast_predlen_{PRED_LEN}.png'
    plt.savefig(filename)
    print(f"\n Plot saved as '{filename}'")




#Weather
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(10)
torch.manual_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

data = df[feature_cols].values.astype(np.float32)
print(f"Dataset loaded: {data.shape}")

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

n = len(data_scaled)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

data_train = data_scaled[:train_end]
data_val = data_scaled[train_end:val_end]
data_test = data_scaled[val_end:]

print(f"Train: {len(data_train)}, Val: {len(data_val)}, Test: {len(data_test)}")

def create_sequences(data, seq_len=48, pred_len=96):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=3):
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
    def __init__(self, embed_dim, num_heads=4, k=3):
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
    
class I3Informer(nn.Module):
    def __init__(self, input_dim=21, embed_dim=32, patch_size=6, pred_len=96, topk=3):
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
    print(f" Training for Prediction Length: {PRED_LEN}")
    
    X_train, y_train = create_sequences(data_train, SEQ_LEN, PRED_LEN)
    X_val, y_val = create_sequences(data_val, SEQ_LEN, PRED_LEN)
    X_test, y_test = create_sequences(data_test, SEQ_LEN, PRED_LEN)
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print(f" Skipping PRED_LEN={PRED_LEN}: insufficient data")
        results_summary[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue
    
    print(f" Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = I3Informer(
        input_dim=len(feature_cols),
        embed_dim=32,
        patch_size=6,
        pred_len=PRED_LEN,
        topk=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    print("\n Training started...\n")
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}/50 → Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)

    print("\n Evaluating on test set...\n")
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
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
    print(f" Results for PRED_LEN={PRED_LEN}:")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    
    last_true = y_true[-1, :, 0]  
    last_pred = y_pred[-1, :, 0]
    last_true_inv = scaler.inverse_transform(
        np.concatenate([last_true.reshape(-1, 1), 
                       np.zeros((len(last_true), F-1))], axis=1)
    )[:, 0]
    last_pred_inv = scaler.inverse_transform(
        np.concatenate([last_pred.reshape(-1, 1), 
                       np.zeros((len(last_pred), F-1))], axis=1)
    )[:, 0]
    
    plt.figure(figsize=(12, 5))
    plt.plot(last_true_inv, label='Actual', marker='o', markersize=3)
    plt.plot(last_pred_inv, label='Predicted', marker='x', markersize=3)
    plt.title(f'Last Test Sequence - Pred Len {PRED_LEN} (Feature: {feature_cols[0]})')
    plt.xlabel('Time Steps')
    plt.ylabel(feature_cols[0])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'weather_i3informer_predlen_{PRED_LEN}.png', dpi=150)
    plt.show()

print(f"\n\n{'='*70}")
print(" I3INFORMER - FINAL RESULTS SUMMARY")
print(f"{'Pred Len':<15} {'MAE':<20} {'RMSE':<20}")
for pred_len in PRED_LENS:
    if results_summary[pred_len]['MAE'] is not None:
        mae = results_summary[pred_len]['MAE']
        rmse = results_summary[pred_len]['RMSE']
        print(f"{pred_len:<15} {mae:<20.4f} {rmse:<20.4f}")
    else:
        print(f"{pred_len:<15} {'SKIPPED':<20} {'SKIPPED':<20}")



# ILINet

import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

print("Loading ILINet data...")
df = pd.read_csv("ILINet.csv", header=None)
df.columns = [
    "REGION TYPE", "REGION", "YEAR", "WEEK", "% WEIGHTED ILI",
    "Original_%UNWEIGHTED ILI", "ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"
]

df["ILITOTAL"] = pd.to_numeric(df["ILITOTAL"], errors="coerce")
df["TOTAL PATIENTS"] = pd.to_numeric(df["TOTAL PATIENTS"], errors="coerce")
df["%UNWEIGHTED ILI"] = (df["ILITOTAL"] / df["TOTAL PATIENTS"]) * 100
df = df.dropna(subset=["%UNWEIGHTED ILI"])

df_agg = df.groupby(['YEAR', 'WEEK']).agg({
    'ILITOTAL': 'sum',
    'TOTAL PATIENTS': 'sum'
}).reset_index()
df_agg['%UNWEIGHTED ILI'] = (df_agg['ILITOTAL'] / df_agg['TOTAL PATIENTS']) * 100
df_agg = df_agg.sort_values(['YEAR', 'WEEK']).reset_index(drop=True)

data = df_agg[['%UNWEIGHTED ILI']].values.astype(np.float32)
print(f" Loaded {len(data)} weekly ILI records")

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, k=3):
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
    def __init__(self, embed_dim, num_heads=4, k=3):
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

class I3Informer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=32, patch_size=6, pred_len=96, topk=3):
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
        T_new = (T // self.patch_size) * self.patch_size
        x = x[:, :T_new, :]
        num_patches = T_new // self.patch_size
        x = x.view(B * num_patches, self.patch_size, -1)
        x = self.local(x)
        x = x.mean(dim=1).view(B, num_patches, -1)
        x = self.global_(x)
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))

results = {}
best_models = {}

for PRED_LEN in PRED_LENS:
    print(f"Processing Prediction Length: {PRED_LEN} weeks")
    
    set_seed(10)
    
    min_required = SEQ_LEN + PRED_LEN + 100
    if len(data) < min_required:
        print(f" Skipping PRED_LEN={PRED_LEN}: insufficient data ({len(data)} < {min_required})")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue
    
    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    
    if len(X_seq) < 50:
        print(f" Skipping PRED_LEN={PRED_LEN}: too few sequences ({len(X_seq)})")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None, 'R2': None}
        continue
    
    print(f"Created {len(X_seq)} sequences")
    
    n = len(X_seq)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    print(f" Split: Train={len(X_train)} ({100*len(X_train)/n:.1f}%), "
          f"Val={len(X_val)} ({100*len(X_val)/n:.1f}%), "
          f"Test={len(X_test)} ({100*len(X_test)/n:.1f}%)")
    
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ), batch_size=32, shuffle=False)
    
    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ), batch_size=32, shuffle=False)
    
    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    ), batch_size=32, shuffle=False)
    
    model = I3Informer(
        input_dim=1, 
        embed_dim=32, 
        patch_size=6, 
        pred_len=PRED_LEN, 
        topk=3
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("\n Training...")
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, 31):
        
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).squeeze(-1)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).squeeze(-1)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}/30 → Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    best_models[PRED_LEN] = best_model_state
    
    print(f"\n Evaluating on test set...")
    model.load_state_dict(best_model_state)
    model.eval()
    
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device).squeeze(-1)
            pred = model(xb).cpu()
            y_true_all.append(yb.cpu())
            y_pred_all.append(pred)
    
    y_true_all = torch.cat(y_true_all, dim=0).numpy().reshape(-1, 1)
    y_pred_all = torch.cat(y_pred_all, dim=0).numpy().reshape(-1, 1)
    
    y_true_inv = scaler.inverse_transform(y_true_all)
    y_pred_inv = scaler.inverse_transform(y_pred_all)
    
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    r2 = r2_score(y_true_inv, y_pred_inv)
    
    results[PRED_LEN] = {
        'MAE': mae, 
        'RMSE': rmse, 
        'R2': r2,
        'best_val_loss': best_val_loss
    }
    
    print(f"\n Test Results for PRED_LEN={PRED_LEN}:")
    print(f"   MAE  = {mae:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   R²   = {r2:.4f}")
    print(f"   Best Val Loss = {best_val_loss:.6f}")

print("I³-INFORMER — ILINET FORECASTING RESULTS (Test Set)")
print(f"{'Pred Len':<12} {'MAE':<15} {'RMSE':<15} {'R² Score':<15}")
for pred_len in PRED_LENS:
    res = results[pred_len]
    if res['MAE'] is not None:
        print(f"{pred_len:<12} {res['MAE']:<15.4f} {res['RMSE']:<15.4f} {res['R2']:<15.4f}")
    else:
        print(f"{pred_len:<12} {'SKIPPED':<15} {'SKIPPED':<15} {'SKIPPED':<15}")


valid_lengths = [pl for pl in PRED_LENS if results[pl]['MAE'] is not None]

if valid_lengths:
    fig, axes = plt.subplots(len(valid_lengths), 1, figsize=(16, 5*len(valid_lengths)))
    if len(valid_lengths) == 1:
        axes = [axes]
    
    for idx, PRED_LEN in enumerate(valid_lengths):
        print(f"\n Creating plot for prediction length {PRED_LEN}...")
        
        set_seed(10)
        
        X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
        n = len(X_seq)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        X_test, y_test = X_seq[val_end:], y_seq[val_end:]
        
        model = I3Informer(
            input_dim=1, 
            embed_dim=32, 
            patch_size=6, 
            pred_len=PRED_LEN, 
            topk=3
        ).to(device)
        model.load_state_dict(best_models[PRED_LEN])
        model.eval()
        
        last_X = X_test[-1:]  
        last_y = y_test[-1:]  
        
        X_plot_tensor = torch.tensor(last_X, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_scaled = model(X_plot_tensor).cpu().numpy().reshape(-1, 1)
        
        y_true_scaled = last_y.reshape(-1, 1)
        y_true_inv = scaler.inverse_transform(y_true_scaled)
        y_pred_inv = scaler.inverse_transform(y_pred_scaled)
        
        context_scaled = last_X.reshape(-1, 1)[-24:]
        context_inv = scaler.inverse_transform(context_scaled)
        
        ax = axes[idx]
        time_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)
        time_context = np.arange(SEQ_LEN - 24, SEQ_LEN)
        
        ax.plot(time_context, context_inv, color='lightgray', 
                label='Context (last 24 weeks)', linewidth=2, alpha=0.7)
        ax.plot(time_pred, y_true_inv, color='#2E86AB', 
                label='Actual ILI %', linewidth=2.5, alpha=0.9)
        ax.plot(time_pred, y_pred_inv, color='#A23B72', linestyle='--',
                label='Predicted ILI %', linewidth=2.5, alpha=0.9)
        ax.axvline(x=SEQ_LEN, color='black', linestyle=':', 
                   alpha=0.7, linewidth=1.5, label='Prediction Start')
        
        ax.set_title(f'ILI Forecasting – Input: {SEQ_LEN} weeks, Forecast: {PRED_LEN} weeks '
                     f'(MAE: {results[PRED_LEN]["MAE"]:.4f}, RMSE: {results[PRED_LEN]["RMSE"]:.4f})',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Week Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Unweighted ILI (%)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("ilinet_all_predictions.png", dpi=150, bbox_inches='tight')
    print(f" Saved: ilinet_all_predictions.png")
    
    print(f"\n Creating comparison chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pred_lens = valid_lengths
    maes = [results[pl]['MAE'] for pl in pred_lens]
    rmses = [results[pl]['RMSE'] for pl in pred_lens]
    
    x = np.arange(len(pred_lens))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, maes, width, label='MAE', 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, rmses, width, label='RMSE', 
                   color='#A23B72', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Prediction Length (weeks)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Value', fontsize=12, fontweight='bold')
    ax.set_title('I³-Informer Performance Across Prediction Lengths (ILINet)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pred_lens)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("ilinet_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: ilinet_comparison.png")
    
    plt.show()

print("\n Experiment completed with proper train/val/test split!")
print(" All visualizations saved successfully!")
print("\n" + "="*80)

# IDEA Stock Forecasting 

import random
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


def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv("IDEA.csv")
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
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=3):
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
    def __init__(self, embed_dim, num_heads=4, top_k=3, dropout=0.1):
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

class I3Informer(nn.Module):
    def __init__(self, input_dim=1, embed_dim=64, num_heads=4, top_k=3, 
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

results = {}

for PRED_LEN in PRED_LENS:
    print(f"Processing Prediction Length: {PRED_LEN}")
    
    set_seed(10)

    min_required = SEQ_LEN + PRED_LEN + 10
    if len(data_scaled) < min_required:
        print(f"Skipping PRED_LEN={PRED_LEN}: insufficient data")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) == 0:
        print(f"Skipping PRED_LEN={PRED_LEN}: no valid sequences")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue
        
    print(f" Created {len(X_seq)} sequences")

    n = len(X_seq)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]

    print(f" Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ), batch_size=32, shuffle=False)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ), batch_size=32, shuffle=False)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    ), batch_size=32, shuffle=False)

    model = I3Informer(
        input_dim=1,
        embed_dim=64,
        num_heads=4,
        top_k=3,
        patch_size=6,
        num_layers=2,
        dropout=0.1,
        pred_len=PRED_LEN
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.MSELoss()

    print("Training...")
    best_val_loss = float('inf')
    best_model_state = None
    losses = []

    for epoch in range(20):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        losses.append(train_loss)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20 - Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

    model.load_state_dict(best_model_state)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(pred)

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    
    results[PRED_LEN] = {
        'MAE': mae,
        'RMSE': rmse,
        'best_model_state': best_model_state,
        'losses': losses,
        'y_true': y_true_inv,
        'y_pred': y_pred_inv,
        'X_test': X_test,
        'y_test': y_test
    }
    print(f" Test Results for PRED_LEN={PRED_LEN}:")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")

print("I³-INFORMER — IDEA STOCK FORECASTING (TEST SET)")

print(f"{'Pred Len':<10} {'MAE':<15} {'RMSE':<15}")
for pred_len in PRED_LENS:
    res = results[pred_len]
    if res['MAE'] is not None:
        print(f"{pred_len:<10} {res['MAE']:<15.4f} {res['RMSE']:<15.4f}")
    else:
        print(f"{pred_len:<10} {'SKIPPED':<15} {'SKIPPED':<15}")

valid_lengths = [pl for pl in PRED_LENS if results[pl]['MAE'] is not None]

for PRED_LEN in valid_lengths:
   
    best_model_state = results[PRED_LEN]['best_model_state']
    X_test = results[PRED_LEN]['X_test']
    y_test = results[PRED_LEN]['y_test']
    
    X_last = torch.tensor(X_test[-1:], dtype=torch.float32).to(device)
    y_last_true = y_test[-1:]

    model = I3Informer(
        input_dim=1,
        embed_dim=64,
        num_heads=4,
        top_k=3,
        patch_size=6,
        num_layers=2,
        dropout=0.1,
        pred_len=PRED_LEN
    ).to(device)
    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_last_pred_scaled = model(X_last).cpu().numpy().reshape(-1, 1)

    y_true_plot = scaler.inverse_transform(y_last_true.reshape(-1, 1))
    y_pred_plot = scaler.inverse_transform(y_last_pred_scaled)

    context_scaled = X_test[-1].reshape(-1, 1)[-24:]
    context_plot = scaler.inverse_transform(context_scaled)

    
    plt.figure(figsize=(14, 5))
    time_context = np.arange(SEQ_LEN - 24, SEQ_LEN)
    time_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)

    plt.plot(time_context, context_plot, 'lightgray', label='Context (last 24 days)', linewidth=1)
    plt.plot(time_pred, y_true_plot, 'b-', label='Actual Close Price (₹)', linewidth=2)
    plt.plot(time_pred, y_pred_plot, 'r--', label='Predicted Close Price (₹)', linewidth=2)
    plt.axvline(x=SEQ_LEN, color='k', linestyle=':', alpha=0.7, label='Prediction Start')
    plt.title(f'IDEA Stock Forecasting (I³-Informer) — Prediction Length = {PRED_LEN} days', fontsize=14)
    plt.xlabel('Day Index', fontsize=12)
    plt.ylabel('Close Price (₹)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f'i3informer_idea_predlen_{PRED_LEN}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f" Plot saved as '{filename}'")

print("\n Experiment completed .")

# Electricity Load Forecasting

import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

print("Loading data...")
df = pd.read_csv("Electricity_load.csv")
if "Unnamed: 0" in df.columns:
    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date")

feature_cols = [c for c in df.columns if c.startswith("MT_")]
data = df[feature_cols].values.astype(np.float32)
num_features = len(feature_cols)
print(f" Loaded {num_features} features (MT_ columns)")
print(f" Total data points: {len(data)}")
print(f" Feature columns: {feature_cols[:5]}... (showing first 5)")

print("\n DATA STATISTICS:")
for i in range(min(3, num_features)):
    col = feature_cols[i]
    vals = data[:, i]
    print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}, std={vals.std():.4f}")

scaler = StandardScaler()  
data_scaled = scaler.fit_transform(data)

print("\n SCALER VERIFICATION:")
print(f"  Scaler mean (first 3 features): {scaler.mean_[:3]}")
print(f"  Scaler scale (first 3 features): {scaler.scale_[:3]}")
print(f"  Original data range for MT_001: [{data[:, 0].min():.4f}, {data[:, 0].max():.4f}]")

test_point = data_scaled[0:1, :]  
test_inverse = scaler.inverse_transform(test_point)
print(f"\n SCALER TEST:")
print(f"  Original first point (MT_001): {data[0, 0]:.4f}")
print(f"  Scaled first point (MT_001): {data_scaled[0, 0]:.4f}")
print(f"  Inverse transformed back (MT_001): {test_inverse[0, 0]:.4f}")
print(f"  Match? {np.allclose(data[0, 0], test_inverse[0, 0])}")

print("\n SCALED DATA STATISTICS:")
for i in range(min(3, num_features)):
    col = feature_cols[i]
    vals = data_scaled[:, i]
    print(f"  {col}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}, std={vals.std():.4f}")

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]
NUM_WORKERS = min(2, mp.cpu_count()) 
print(f"\n🔧 Using {NUM_WORKERS} DataLoader workers for parallel data loading")

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=3):
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
        k = min(self.top_k, L)
        top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
        
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, top_idx, top_vals)
        attn = torch.softmax(masked_logits, dim=-1)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, E)
        return self.out(out)

class InformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=3):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, top_k)
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

class I3Informer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, patch_size=6, pred_len=96, top_k=3):
        super().__init__()
        self.patch_size = patch_size
        self.pred_len = pred_len
        self.input_dim = input_dim
        
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
        return out.view(B, self.pred_len, self.input_dim)

def train_model_for_pred_len(args):
    """Train model for a specific prediction length"""
    PRED_LEN, data_scaled, num_features, device_id, scaler_obj, feature_cols = args
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    print(f" Processing PRED_LEN={PRED_LEN} on {device}")
    
    set_seed(10)
    
    min_required = SEQ_LEN + PRED_LEN + 100
    if len(data_scaled) < min_required:
        print(f" Skipping PRED_LEN={PRED_LEN}: insufficient data")
        return PRED_LEN, {'MAE': None, 'RMSE': None, 'R2': None}, None
    
    X_seq, y_seq = create_sequences(data_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) < 100:
        print(f" Skipping PRED_LEN={PRED_LEN}: too few sequences")
        return PRED_LEN, {'MAE': None, 'RMSE': None, 'R2': None}, None
    
    print(f" Created {len(X_seq)} sequences")
    
    n = len(X_seq)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    print(f" Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    print(f"\n TEST DATA RANGE CHECK:")
    print(f"  Test sequences: {len(X_test)}")
    print(f"  Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    total_sequences = len(X_seq)
    test_start_idx = val_end
    print(f"  Test uses sequences from index {test_start_idx} to {total_sequences}")
    
    first_test_seq_start = test_start_idx
    first_test_pred_start = first_test_seq_start + SEQ_LEN
    first_test_pred_end = first_test_pred_start + PRED_LEN
    
    print(f"\n CHECKING ORIGINAL DATA FOR FIRST TEST SEQUENCE:")
    print(f"  First test sequence uses input from data[{first_test_seq_start}:{first_test_seq_start+SEQ_LEN}]")
    print(f"  And predicts data[{first_test_pred_start}:{first_test_pred_end}]")
    print(f"  Total data length: {len(data_scaled)}")
    
    if first_test_pred_end <= len(data_scaled):
        original_target = data_scaled[first_test_pred_start:first_test_pred_end, 0]  
        print(f"  Original scaled target (first {min(10, len(original_target))} points, feature 0): {original_target[:10]}")
        print(f"  Stats: min={original_target.min():.4f}, max={original_target.max():.4f}, std={original_target.std():.4f}")
    
    for i in range(min(3, num_features)):
        col = feature_cols[i]
        test_vals = y_test[:, :, i].flatten()
        print(f"\n  {col} in y_test (scaled targets):")
        print(f"    Shape: {y_test[:, :, i].shape}")
        print(f"    First 10 values: {test_vals[:10]}")
        print(f"    Stats: min={test_vals.min():.4f}, max={test_vals.max():.4f}, std={test_vals.std():.4f}")
        print(f"    Unique values: {len(np.unique(test_vals))}, All same? {len(np.unique(test_vals)) == 1}")
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ), 
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ), 
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        ), 
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n VERIFYING TEST DATALOADER:")
    test_iter = iter(test_loader)
    first_batch_x, first_batch_y = next(test_iter)
    print(f"  First batch shapes: X={first_batch_x.shape}, y={first_batch_y.shape}")
    print(f"  First batch y (feature 0, first sample, first 10 timesteps): {first_batch_y[0, :10, 0]}")
    print(f"  Stats of first batch y (feature 0): min={first_batch_y[:, :, 0].min():.4f}, max={first_batch_y[:, :, 0].max():.4f}")
    
    model = I3Informer(
        input_dim=num_features,
        embed_dim=128,
        patch_size=6,
        pred_len=PRED_LEN,
        top_k=3
    ).to(device)
    
    scaler_amp = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(" Training with mixed precision...")
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(20):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            
            if scaler_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                optimizer.zero_grad()
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                
                if scaler_amp:
                    with torch.amp.autocast('cuda'):
                        pred = model(xb)
                        loss = criterion(pred, yb)
                else:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/20 - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(" Evaluating on test set...")
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    
    y_true_all, y_pred_all = [], []
    batch_count = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            if scaler_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(xb).cpu()
            else:
                pred = model(xb).cpu()
            
            if batch_count == 0:
                print(f"\n FIRST BATCH EVALUATION:")
                print(f"  Input shape: {xb.shape}")
                print(f"  True target shape: {yb.shape}")
                print(f"  Prediction shape: {pred.shape}")
                print(f"  True values (first sample, first 10 steps, feature 0): {yb[0, :10, 0]}")
                print(f"  Pred values (first sample, first 10 steps, feature 0): {pred[0, :10, 0]}")
            
            y_true_all.append(yb)
            y_pred_all.append(pred)
            batch_count += 1
    
    y_true = torch.cat(y_true_all, dim=0).numpy()
    y_pred = torch.cat(y_pred_all, dim=0).numpy()
    
    print(f"\n BEFORE INVERSE TRANSFORM:")
    print(f"  y_true shape: {y_true.shape}")  
    print(f"  y_pred shape: {y_pred.shape}")
    
    print(f"\n RAW SCALED VALUES (first sequence, first 10 timesteps, feature 0):")
    print(f"  y_true scaled: {y_true[0, :10, 0]}")
    print(f"  y_pred scaled: {y_pred[0, :10, 0]}")
    
    num_sequences = y_true.shape[0]
    y_true_flat = y_true.reshape(-1, num_features)
    y_pred_flat = y_pred.reshape(-1, num_features)
    
    print(f"  y_true_flat shape: {y_true_flat.shape}")
    print(f"  y_pred_flat shape: {y_pred_flat.shape}")
    
    print(f"\n FLATTENED VALUES (first 10 points, feature 0):")
    print(f"  y_true_flat: {y_true_flat[:10, 0]}")
    print(f"  y_pred_flat: {y_pred_flat[:10, 0]}")
    
    y_true_inv = scaler_obj.inverse_transform(y_true_flat)
    y_pred_inv = scaler_obj.inverse_transform(y_pred_flat)
    
    print(f"\n AFTER INVERSE TRANSFORM (first 10 points, feature 0):")
    print(f"  y_true_inv: {y_true_inv[:10, 0]}")
    print(f"  y_pred_inv: {y_pred_inv[:10, 0]}")
    
    print(f"\n AFTER INVERSE TRANSFORM:")
    for i in range(min(3, num_features)):
        col = feature_cols[i]
        true_vals = y_true_inv[:, i]
        pred_vals = y_pred_inv[:, i]
        print(f"  {col}:")
        print(f"    True:  min={true_vals.min():.4f}, max={true_vals.max():.4f}, std={true_vals.std():.4f}")
        print(f"    Pred:  min={pred_vals.min():.4f}, max={pred_vals.max():.4f}, std={pred_vals.std():.4f}")
        print(f"    First 10 true values: {true_vals[:10]}")
        print(f"    Are all same? {len(np.unique(true_vals)) == 1}")
    
    mae_per_var = np.mean(np.abs(y_true_inv - y_pred_inv), axis=0)
    rmse_per_var = np.sqrt(np.mean((y_true_inv - y_pred_inv) ** 2, axis=0))
    
    avg_mae = np.mean(mae_per_var)
    avg_rmse = np.mean(rmse_per_var)
    r2 = r2_score(y_true_flat, y_pred_flat, multioutput='variance_weighted')
    
    results = {
        'MAE': avg_mae,
        'RMSE': avg_rmse,
        'R2': r2,
        'best_val_loss': best_val_loss,
        'y_true_inv': y_true_inv,
        'y_pred_inv': y_pred_inv,
        'y_true_scaled': y_true,
        'y_pred_scaled': y_pred
    }
    
    print(f"\n Test Results for PRED_LEN={PRED_LEN}:")
    print(f"   Avg MAE  = {avg_mae:.4f}")
    print(f"   Avg RMSE = {avg_rmse:.4f}")
    print(f"   R² Score = {r2:.4f}\n")
    
    print(f" Saving predictions to CSV...")
    pred_df = pd.DataFrame({
        'timestep': np.arange(len(y_true_inv))
    })
    
    for i, col in enumerate(feature_cols):
        pred_df[f'{col}_true'] = y_true_inv[:, i]
        pred_df[f'{col}_predicted'] = y_pred_inv[:, i]
    
    csv_filename = f"predictions_predlen_{PRED_LEN}.csv"
    pred_df.to_csv(csv_filename, index=False)
    print(f"   Saved to: {csv_filename}")
    print(f"   Shape: {pred_df.shape}")
    print(f"   Columns: {list(pred_df.columns[:5])}... (showing first 5)")
    
    return PRED_LEN, results, best_model_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"\n Using device: {device}")
print(f" Available GPUs: {num_gpus}\n")

results = {}
best_models = {}

if num_gpus > 1:
    print(f" MULTI-GPU MODE: Training {len(PRED_LENS)} models in parallel")
    
    args_list = [
        (PRED_LEN, data_scaled, num_features, i % num_gpus, scaler, feature_cols)
        for i, PRED_LEN in enumerate(PRED_LENS)
    ]
    
    with ThreadPoolExecutor(max_workers=min(len(PRED_LENS), num_gpus)) as executor:
        futures = [executor.submit(train_model_for_pred_len, args) for args in args_list]
        
        for future in futures:
            pred_len, result, model_state = future.result()
            results[pred_len] = result
            if model_state is not None:
                best_models[pred_len] = model_state

else:
    print("SINGLE DEVICE MODE: Training models sequentially with optimizations")
    
    for PRED_LEN in PRED_LENS:
        pred_len, result, model_state = train_model_for_pred_len(
            (PRED_LEN, data_scaled, num_features, 0, scaler, feature_cols)
        )
        results[pred_len] = result
        if model_state is not None:
            best_models[pred_len] = model_state

print("I³-INFORMER — FINAL TEST RESULTS SUMMARY")
print(f"{'Pred Len':<12} {'Avg MAE':<15} {'Avg RMSE':<15} {'R² Score':<15}")
for pred_len in PRED_LENS:
    res = results[pred_len]
    if res['MAE'] is not None:
        print(f"{pred_len:<12} {res['MAE']:<15.4f} {res['RMSE']:<15.4f} {res['R2']:<15.4f}")
    else:
        print(f"{pred_len:<12} {'SKIPPED':<15} {'SKIPPED':<15} {'SKIPPED':<15}")

print("GENERATING VISUALIZATIONS")

valid_lengths = [pl for pl in PRED_LENS if results[pl]['MAE'] is not None]

if valid_lengths:
    PRED_LEN = valid_lengths[0]
    print(f"\n Creating visualization for prediction length {PRED_LEN}...")
    
    
    y_true_inv = results[PRED_LEN]['y_true_inv']
    y_pred_inv = results[PRED_LEN]['y_pred_inv']
    
    print(f"\n DATA FOR PLOTTING:")
    print(f"  y_true_inv shape: {y_true_inv.shape}")
    print(f"  y_pred_inv shape: {y_pred_inv.shape}")
    
    features_to_plot = min(3, num_features)
    fig, axes = plt.subplots(features_to_plot, 1, figsize=(16, 4*features_to_plot))
    if features_to_plot == 1:
        axes = [axes]
    
    plot_points = min(500, len(y_true_inv))
    
    for idx in range(features_to_plot):
        ax = axes[idx]
        true_vals = y_true_inv[:plot_points, idx]
        pred_vals = y_pred_inv[:plot_points, idx]
        
        print(f"\n PLOTTING {feature_cols[idx]}:")
        print(f"  True values:  min={true_vals.min():.4f}, max={true_vals.max():.4f}, std={true_vals.std():.4f}")
        print(f"  Pred values:  min={pred_vals.min():.4f}, max={pred_vals.max():.4f}, std={pred_vals.std():.4f}")
        print(f"  First 10 true: {true_vals[:10]}")
        print(f"  Unique true values: {len(np.unique(true_vals))}")
        
        ax.plot(true_vals, label='True', linewidth=2, alpha=0.8, color='#2E86AB')
        ax.plot(pred_vals, label='Predicted', linewidth=2, alpha=0.8, color='#A23B72')
        
        ax.set_title(f'{feature_cols[idx]} - Test Set (Pred Length: {PRED_LEN})', 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=11)
        ax.set_ylabel('Load Value', fontsize=11)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    filename = f"i3_informer_debug_predlen_{PRED_LEN}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {filename}")
    
    if len(valid_lengths) > 1:
        print(f"\n Creating comparison plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        pred_lens = [pl for pl in valid_lengths]
        maes = [results[pl]['MAE'] for pl in pred_lens]
        rmses = [results[pl]['RMSE'] for pl in pred_lens]
        
        x = np.arange(len(pred_lens))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, maes, width, label='MAE', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, rmses, width, label='RMSE', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Prediction Length', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Value', fontsize=12, fontweight='bold')
        ax.set_title('I³-Informer Performance (with Debugging)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pred_lens)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig("i3_informer_comparison_debug.png", dpi=150, bbox_inches='tight')
        print(f" Saved: i3_informer_comparison_debug.png")

print("\n Debug experiment completed!")
print(" All visualizations saved successfully!")
print(" Check console output for detailed data statistics!")

for pred_len in valid_lengths:
    if results[pred_len]['MAE'] is not None:
        y_true_inv = results[pred_len]['y_true_inv']
        y_pred_inv = results[pred_len]['y_pred_inv']
        
        detailed_df = pd.DataFrame()
        detailed_df['timestep'] = np.arange(len(y_true_inv))
        
        for i, col in enumerate(feature_cols):
            detailed_df[f'{col}_actual'] = y_true_inv[:, i]
            detailed_df[f'{col}_predicted'] = y_pred_inv[:, i]
            detailed_df[f'{col}_error'] = np.abs(y_true_inv[:, i] - y_pred_inv[:, i])
        
       
        csv_name = f"detailed_predictions_predlen_{pred_len}.csv"
        detailed_df.to_csv(csv_name, index=False)
        print(f" Saved: {csv_name} ({len(detailed_df)} rows, {len(detailed_df.columns)} columns)")
        
        summary_df = detailed_df[['timestep', f'{feature_cols[0]}_actual', 
                                   f'{feature_cols[0]}_predicted', f'{feature_cols[0]}_error']].head(100)
        summary_name = f"summary_first_100_predlen_{pred_len}.csv"
        summary_df.to_csv(summary_name, index=False)
        print(f" Saved summary: {summary_name}")


#ETTm1
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

np.random.seed(10)
torch.manual_seed(10)

df = pd.read_csv("ETTm1.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

input_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
target_col = "OT"

X_data = df[input_cols].values.astype(np.float32)
y_data = df[target_col].values.astype(np.float32).reshape(-1, 1)


scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(y_data)

print(" DATA STATISTICS:")
print(f"  Original OT range: [{y_data.min():.2f}, {y_data.max():.2f}]")
print(f"  Scaled OT range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")
print(f"  Input features scaled: {X_scaled.shape}")

n = len(y_scaled)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

X_train, y_train = X_scaled[:train_end], y_scaled[:train_end]
X_val, y_val = X_scaled[train_end:val_end], y_scaled[train_end:val_end]
X_test, y_test = X_scaled[val_end:], y_scaled[val_end:]

def create_sequences(X, y, seq_len=48, pred_len=96):
    X_seq, y_seq = [], []
    for i in range(len(y) - seq_len - pred_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(X_seq), np.array(y_seq)

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=3):
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

        Q = self.q(Q).view(B, L, H, Dh).transpose(1, 2)
        K = self.k(K).view(B, L, H, Dh).transpose(1, 2)
        V = self.v(V).view(B, L, H, Dh).transpose(1, 2)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Dh)

        top_vals, top_idx = torch.topk(logits, k=self.top_k, dim=-1)

        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, top_idx, top_vals)

        attn = torch.softmax(masked_logits, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, E)
        return self.out(out)

class InformerBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, top_k=3):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, num_heads, top_k)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x

class Twinformer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, patch_size=6,
                 pred_len=96, top_k=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Linear(input_dim, embed_dim)

        self.local = InformerBlock(embed_dim, top_k=top_k)
        self.global_ = InformerBlock(embed_dim, top_k=top_k)

        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, pred_len)  

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
        return out.view(B, -1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n Using:", device)

results = {}

for PRED_LEN in PRED_LENS:
    print(f"Twinformer - Processing Pred Len: {PRED_LEN}")
    
    X_tr, y_tr = create_sequences(X_train, y_train, SEQ_LEN, PRED_LEN)
    X_v, y_v = create_sequences(X_val, y_val, SEQ_LEN, PRED_LEN)
    X_te, y_te = create_sequences(X_test, y_test, SEQ_LEN, PRED_LEN)
    
    if len(X_tr) == 0 or len(X_v) == 0 or len(X_te) == 0:
        print(f" Skipping {PRED_LEN}: insufficient data")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue
    
    print(f" Created {len(X_tr)} train, {len(X_v)} val, {len(X_te)} test sequences")
    
    print(f" Sequence statistics:")
    print(f"  y_train range: [{y_tr.min():.4f}, {y_tr.max():.4f}]")
    print(f"  y_test range: [{y_te.min():.4f}, {y_te.max():.4f}]")
    
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), 
                                  torch.tensor(y_tr, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_v, dtype=torch.float32), 
                                torch.tensor(y_v, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_te, dtype=torch.float32), 
                                 torch.tensor(y_te, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = Twinformer(
        input_dim=len(input_cols),
        embed_dim=64,
        patch_size=6,
        pred_len=PRED_LEN,
        top_k=3
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    print(" Training...")
    for epoch in range(50):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_model_state)
    
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).cpu().numpy()
            y_true.append(yb.numpy())
            y_pred.append(pred)
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    yt_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1))
    yp_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    
    print(f"\n After inverse transform:")
    print(f"  True range: [{yt_inv.min():.2f}, {yt_inv.max():.2f}]")
    print(f"  Pred range: [{yp_inv.min():.2f}, {yp_inv.max():.2f}]")
    
    mae = np.mean(np.abs(yt_inv - yp_inv))
    rmse = np.sqrt(np.mean((yt_inv - yp_inv)**2))
    results[PRED_LEN] = {'MAE': mae, 'RMSE': rmse}
    
    print(f"\n Results for PRED_LEN={PRED_LEN}:")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")

for pred_len in PRED_LENS:
    if results[pred_len]['MAE'] is not None:
        mae = results[pred_len]['MAE']
        rmse = results[pred_len]['RMSE']
        print(f"{pred_len:<20} {mae:<20.4f} {rmse:<20.4f}")
    else:
        print(f"{pred_len:<20} {'SKIPPED':<20} {'SKIPPED':<20}")



# ETTh1 Forecasting

import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def set_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(10)

df = pd.read_csv("ETTh1.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

input_features = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
target_col = "OT"

X_data = df[input_features].values.astype(np.float32)  
y_data = df[[target_col]].values.astype(np.float32)    

print(f" Input features: {input_features}")
print(f" Target: {target_col}")
print(f" Shapes: X={X_data.shape}, y={y_data.shape}")

y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y_data)

X_scaled = X_data  

SEQ_LEN = 48
PRED_LENS = [96, 120, 336, 720]

def create_sequences(X, y, seq_len, pred_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len - pred_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(X_seq), np.array(y_seq)


class TopKSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, top_k=3):
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
    def __init__(self, embed_dim=64, num_heads=4, top_k=3):
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

class I3Informer(nn.Module):
    def __init__(self, input_dim, embed_dim=64, patch_size=6, pred_len=96, top_k=3):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Linear(input_dim, embed_dim)
        self.local = InformerBlock(embed_dim, top_k=top_k)
        self.global_ = InformerBlock(embed_dim, top_k=top_k)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, pred_len) 

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
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

results = {}

for PRED_LEN in PRED_LENS:
    print(f"Processing Prediction Length: {PRED_LEN}")
    
    
    set_seed(10)

    min_required = SEQ_LEN + PRED_LEN + 10
    if len(X_scaled) < min_required:
        print(f" Skipping PRED_LEN={PRED_LEN}: insufficient data")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN, PRED_LEN)
    if len(X_seq) == 0:
        print(f" Skipping PRED_LEN={PRED_LEN}: no valid sequences")
        results[PRED_LEN] = {'MAE': None, 'RMSE': None}
        continue
        
    print(f" Created {len(X_seq)} sequences")

    n = len(X_seq)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]

    print(f" Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).squeeze(-1)
    ), batch_size=32, shuffle=False)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).squeeze(-1)
    ), batch_size=32, shuffle=False)

    test_loader = DataLoader(TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).squeeze(-1)
    ), batch_size=32, shuffle=False)

    model = I3Informer(
        input_dim=len(input_features),
        embed_dim=64,
        patch_size=6,
        pred_len=PRED_LEN,
        top_k=3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Training...")
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(20):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) in [5, 10, 15, 20]:
            print(f"  Epoch {epoch+1}/20 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    model.load_state_dict(best_model_state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(pred)

    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true_inv = y_scaler.inverse_transform(y_true)
    y_pred_inv = y_scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))

    results[PRED_LEN] = {
        'MAE': mae,
        'RMSE': rmse,
        'best_model_state': best_model_state,
        'X_test': X_test,
        'y_test': y_test
    }
    print(f" Test Results for PRED_LEN={PRED_LEN}:")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")

print("I³-INFORMER — ETTh1 (OT Forecasting, Test Set)")
print(f"{'Pred Len':<10} {'MAE':<15} {'RMSE':<15}")
for pred_len in PRED_LENS:
    res = results[pred_len]
    if res['MAE'] is not None:
        print(f"{pred_len:<10} {res['MAE']:<15.4f} {res['RMSE']:<15.4f}")
    else:
        print(f"{pred_len:<10} {'SKIPPED':<15} {'SKIPPED':<15}")

valid_lengths = [pl for pl in PRED_LENS if results[pl]['MAE'] is not None]

for PRED_LEN in valid_lengths:
    best_model_state = results[PRED_LEN]['best_model_state']
    X_test = results[PRED_LEN]['X_test']
    y_test = results[PRED_LEN]['y_test']
    
    X_last = torch.tensor(X_test[-1:], dtype=torch.float32).to(device)
    y_last_true = y_test[-1:]  

    model = I3Informer(
        input_dim=len(input_features),
        embed_dim=64,
        patch_size=6,
        pred_len=PRED_LEN,
        top_k=3
    ).to(device)
    model.load_state_dict(best_model_state)

    with torch.no_grad():
        y_last_pred_scaled = model(X_last).cpu().numpy().reshape(-1, 1)

    y_true_plot = y_scaler.inverse_transform(y_last_true.reshape(-1, 1))
    y_pred_plot = y_scaler.inverse_transform(y_last_pred_scaled)

    
    plt.figure(figsize=(14, 5))
    time_pred = np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN)
    plt.plot(time_pred, y_true_plot, 'b-', label='Actual OT', linewidth=2)
    plt.plot(time_pred, y_pred_plot, 'r--', label='Predicted OT', linewidth=2)
    plt.axvline(x=SEQ_LEN, color='k', linestyle=':', alpha=0.7)
    plt.title(f'ETTh1 Oil Temperature Forecasting (I³-Informer) — Pred Len = {PRED_LEN}')
    plt.xlabel('Time Step')
    plt.ylabel('OT (Oil Temperature)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f'i3informer_etth1_ot_predlen_{PRED_LEN}.png'
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f" Plot saved as '{filename}'")

print("\n Experiment completed.")
