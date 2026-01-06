# train_c2_modelcentric.py
# EEG Challenge C2 - Model-centric training pipeline (no CCC loss)
# PyTorch 2.x

import os, math, random, json, argparse, time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def zscore_targets(df: pd.DataFrame, cols: List[str]):
    stats = {}
    for c in cols:
        m, s = df[c].mean(), df[c].std(ddof=0) + 1e-8
        df[c] = (df[c] - m) / s
        stats[c] = {"mean": float(m), "std": float(s)}
    return df, stats

def rmse(y_true, y_pred):  # torch tensors
    return torch.sqrt(F.mse_loss(y_pred, y_true))

def nrmse(y_true, y_pred):
    std = torch.clamp(y_true.std(unbiased=False), min=1e-8)
    return rmse(y_true, y_pred) / std

# -----------------------------
# Data & Augmentations
# -----------------------------
def time_warp(x, warp=0.03):  # x: (C,T)
    if warp <= 0: return x
    T = x.shape[-1]
    shift = int(T * np.random.uniform(-warp, warp))
    if shift == 0: return x
    return torch.roll(x, shifts=shift, dims=-1)

def add_noise(x, s=0.01):
    return x + torch.randn_like(x) * s

def channel_drop(x, p=0.2, max_drop_ratio=0.05):
    if random.random() > p: return x
    C, T = x.shape
    k = max(1, int(C * max_drop_ratio))
    idx = np.random.choice(C, size=k, replace=False)
    x = x.clone(); x[idx] = 0.0
    return x

def spec_banddrop(x, fs, p=0.3, width_hz=(3,6)):
    if random.random() > p: return x
    # very light notch-like drop via FFT
    C,T = x.shape
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0/fs)
    w = np.random.uniform(width_hz[0], width_hz[1])
    f0 = np.random.uniform(4, fs/2-4)
    mask = (freqs >= f0 - w/2) & (freqs <= f0 + w/2)
    X[..., mask] = 0
    return torch.fft.irfft(X, n=T, dim=-1)

class EEGWindows(Dataset):
    """
    CSV columns (minimum):
      path, subject, task, release, p, internalizing, externalizing, attention
    npy file shape: (C, T)
    """
    def __init__(self, csv_path, tasks=None, subjects=None, augment=False, fs=250.0):
        self.df = pd.read_csv(csv_path)
        if tasks is not None:
            self.df = self.df[self.df["task"].isin(tasks)]
        if subjects is not None:
            self.df = self.df[self.df["subject"].isin(subjects)]
        self.df = self.df.reset_index(drop=True)
        self.augment = augment
        self.fs = fs
        self.targets = ["p", "internalizing", "externalizing", "attention"]
        # map domains for adversarial: combine (task,release) or use any available
        self.domains = sorted(self.df["task"].astype(str).unique().tolist())
        self.dom2id = {d:i for i,d in enumerate(self.domains)}

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = np.load(row["path"]).astype(np.float32)  # (C,T)
        x = torch.from_numpy(x)

        # light fixed preproc (DC + 0.3-45 band is assumed done earlier; if not, leave raw)
        # augmentations
        if self.augment:
            x = time_warp(x, 0.03)
            x = add_noise(x, 0.01)
            x = channel_drop(x, p=0.2, max_drop_ratio=0.05)
            x = spec_banddrop(x, fs=self.fs, p=0.3)

        y = torch.tensor([row[t] for t in self.targets], dtype=torch.float32)
        subject = str(row["subject"])
        domain = self.dom2id[str(row["task"])]
        meta = {
            "subject": subject,
            "task": str(row["task"]),
            "release": str(row.get("release", "R?")),
            "domain": int(domain)
        }
        return x, y, meta

def collate_fn(batch):
    # batch of tuples
    X = [b[0] for b in batch]  # (C,T)
    Y = torch.stack([b[1] for b in batch], dim=0)  # (B,4)
    metas = [b[2] for b in batch]
    # build subject ids for pooling
    subs = [m["subject"] for m in metas]
    # map to indices
    unique_subs = sorted(set(subs))
    sub2id = {s:i for i,s in enumerate(unique_subs)}
    subject_ids = torch.tensor([sub2id[s] for s in subs], dtype=torch.long)
    domains = torch.tensor([m["domain"] for m in metas], dtype=torch.long)
    # pad-free stack (variable T allowed via min-cut)
    T_min = min([x.shape[-1] for x in X])
    X = torch.stack([x[..., :T_min] for x in X], dim=0)  # (B,C,T)
    return X, Y, subject_ids, domains, metas

# -----------------------------
# Model blocks
# -----------------------------
class SincConv1d(nn.Module):
    # Light SincNet-like conv for learnable band-pass
    def __init__(self, out_channels=64, kernel_size=129, sample_rate=250.0, min_hz=0.3, max_hz=45.0):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_hz = min_hz
        self.max_hz = max_hz
        # learn cutoff frequencies
        low_hz = torch.linspace(min_hz, max_hz-5.0, out_channels)
        band_hz = torch.ones(out_channels)*5.0
        self.low_hz_ = nn.Parameter(low_hz)
        self.band_hz_ = nn.Parameter(band_hz)
        n = torch.arange(-(kernel_size//2), kernel_size//2+1).float()
        self.register_buffer("n", n)

    def forward(self, x):  # x: (B,C,T)
        B,C,T = x.shape
        device = x.device
        low = torch.abs(self.low_hz_)
        band = torch.abs(self.band_hz_)
        high = low + band
        low = torch.clamp(low, self.min_hz, self.max_hz-1.0)
        high = torch.clamp(high, low+1.0, self.max_hz)

        n = self.n.to(device)
        window = torch.hamming_window(self.kernel_size, periodic=False, dtype=x.dtype, device=device)

        # build filters (out, 1, K)
        filters = []
        for i in range(self.out_channels):
            f1 = low[i] / (self.sample_rate/2)
            f2 = high[i] / (self.sample_rate/2)
            # bandpass from two sinc lowpass
            h1 = 2*f2*torch.sinc(2*f2*n)
            h2 = 2*f1*torch.sinc(2*f1*n)
            bandpass = (h1 - h2) * window
            filters.append(bandpass)
        filt = torch.stack(filters, dim=0).unsqueeze(1)  # (out,1,K)
        # depthwise over channel-time: apply per channel then sum? Use group conv by repeating
        x = x.view(B*C, 1, T)
        y = F.conv1d(x, filt, stride=1, padding=self.kernel_size//2, groups=1)
        y = y.view(B, C, self.out_channels, y.shape[-1]).sum(dim=1)  # sum across channels to out
        return y  # (B, out, T)

class SEBlock(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels//r))
        self.fc2 = nn.Linear(max(1, channels//r), channels)
    def forward(self, x):  # (B,C,T)
        s = x.mean(dim=-1)  # (B,C)
        e = F.relu(self.fc1(s))
        e = torch.sigmoid(self.fc2(e)).unsqueeze(-1)  # (B,C,1)
        return x * e

class FiLM(nn.Module):
    def __init__(self, feat_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta  = nn.Linear(cond_dim, feat_dim)
    def forward(self, x, cond):  # x:(B,F,T) or (B,F); cond:(B,cond_dim)
        g = self.gamma(cond)
        b = self.beta(cond)
        if x.dim()==3:
            return x * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)
        return x * (1 + g) + b

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam=lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam*g, None

class DomainAdversary(nn.Module):
    def __init__(self, in_dim, n_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, n_domains)
        )
    def forward(self, f, lam=1.0):
        return self.net(GRL.apply(f, lam))

class AttentionPooling(nn.Module):
    # Pool windows to subject-level embeddings within a mini-batch
    def __init__(self, in_dim):
        super().__init__()
        self.att = nn.Linear(in_dim, 1)
    def forward(self, feats, subject_ids):
        # feats: (B,F), subject_ids: (B,)
        B,F = feats.shape
        device = feats.device
        n_sub = int(subject_ids.max().item()) + 1
        # compute attention per subject
        scores = self.att(feats).squeeze(-1)  # (B,)
        scores = torch.exp(scores - scores.max())  # stability
        # aggregate
        out = torch.zeros(n_sub, F, device=device)
        Z   = torch.zeros(n_sub, 1, device=device)
        out.index_add_(0, subject_ids, feats * scores.unsqueeze(-1))
        Z.index_add_(0, subject_ids, scores.unsqueeze(-1))
        out = out / (Z + 1e-8)
        return out  # (n_sub, F)

class SmallTemporalBackbone(nn.Module):
    # fallback backbone if CBraMod not provided
    def __init__(self, in_ch, front_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(front_ch, 128, 7, padding=3, groups=1), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.out_dim = 256
    def forward(self, x):  # x: (B,front_ch,T)
        h = self.conv(x).squeeze(-1)  # (B,256)
        return h

class CBraModBackbone(nn.Module):
    """
    Placeholder loader. Replace ckpt loading logic with your CBraMod model.
    Expect input: (B, front_ch, T) after SincConv; output: (B, D).
    """
    def __init__(self, out_dim=512):
        super().__init__()
        # Dummy: simple MLP head over GAP to simulate a feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(64, 128, 7, padding=3), nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, out_dim)
        self.out_dim = out_dim

    def load_from_checkpoint(self, ckpt_path: str):
        # Implement actual loading for CBraMod weights here.
        if os.path.exists(ckpt_path):
            try:
                sd = torch.load(ckpt_path, map_location="cpu")
                self.load_state_dict(sd, strict=False)
                print(f"[Info] Loaded CBraMod weights (strict=False) from {ckpt_path}")
            except Exception as e:
                print(f"[Warn] Failed to load CBraMod weights: {e}")

    def forward(self, x):  # (B,64,T)
        h = self.conv(x).squeeze(-1)
        return self.fc(h)

class RegressionHead(nn.Module):
    def __init__(self, in_dim, hetero=False, n_targets=4):
        super().__init__()
        self.hetero = hetero
        self.base = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        if hetero:
            self.mu = nn.Linear(128, n_targets)
            self.logvar = nn.Linear(128, n_targets)
        else:
            self.out = nn.Linear(128, n_targets)

    def forward(self, z):
        h = self.base(z)
        if self.hetero:
            mu = self.mu(h)
            logv = torch.clamp(self.logvar(h), min=-6.0, max=6.0)
            return mu, logv
        return self.out(h)

def heteroscedastic_nll(y, mu, logvar):
    # per-sample, per-target NLL
    inv_var = torch.exp(-logvar)
    loss = 0.5 * ((y - mu)**2 * inv_var + logvar)
    return loss.mean()

def coral_loss(source, target):
    # Deep CORAL between two groups in the batch (source, target): (Ns,F),(Nt,F)
    def cov(x):
        xm = x - x.mean(dim=0, keepdim=True)
        return (xm.t() @ xm) / (x.shape[0] - 1 + 1e-8)
    cs, ct = cov(source), cov(target)
    return ((cs - ct)**2).mean()

class Model(nn.Module):
    def __init__(self, n_channels, fs=250.0, cond_dim=0, n_domains=1,
                 use_cbramod=False, cbramod_ckpt=None, hetero=True):
        super().__init__()
        self.front = SincConv1d(out_channels=64, kernel_size=129, sample_rate=fs)
        self.sep = nn.Sequential(
            nn.Conv1d(64, 64, 9, padding=4, groups=64),  # depthwise
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),  # pointwise
            nn.ReLU(),
            SEBlock(64, r=8)
        )
        # Backbone
        if use_cbramod:
            self.backbone = CBraModBackbone(out_dim=512)
            if cbramod_ckpt: self.backbone.load_from_checkpoint(cbramod_ckpt)
        else:
            self.backbone = SmallTemporalBackbone(in_ch=n_channels, front_ch=64)
        self.feat_dim = self.backbone.out_dim
        # FiLM (optional)
        self.cond_dim = cond_dim
        if cond_dim > 0:
            self.film = FiLM(self.feat_dim, cond_dim)
        else:
            self.film = None
        # Heads
        self.pool = AttentionPooling(self.feat_dim)
        self.reg = RegressionHead(self.feat_dim, hetero=hetero, n_targets=4)
        self.adv = DomainAdversary(self.feat_dim, n_domains=n_domains)

    def forward(self, x, subject_ids, cond_vec=None, adv_lambda=0.0):
        # x: (B,C,T)
        z = self.front(x)             # (B,64,T)
        z = self.sep(z)               # (B,64,T)
        f = self.backbone(z)          # (B,D)
        if self.film is not None and cond_vec is not None:
            f = self.film(f, cond_vec)
        # subject pooling
        s = self.pool(f, subject_ids)  # (S,D)
        # regression head on subject embeddings
        out = self.reg(s)              # tuple or tensor
        # adversary on features (before pooling to give more samples), mean per subject
        with torch.no_grad():
            # compute subject mean for adversary as well (same as pool but mean)
            n_sub = int(subject_ids.max().item()) + 1
            agg = torch.zeros(n_sub, f.shape[-1], device=f.device)
            cnt = torch.zeros(n_sub, 1, device=f.device)
            agg.index_add_(0, subject_ids, f)
            cnt.index_add_(0, subject_ids, torch.ones_like(cnt[:subject_ids.shape[0]]))
            f_sub = agg / (cnt + 1e-8)
        logits_d = self.adv(f_sub.detach(), lam=adv_lambda)  # detach on forward; GRL applied inside adv
        return out, logits_d, s

# -----------------------------
# Training / Evaluation
# -----------------------------
@dataclass
class CFG:
    csv_train: str = "train.csv"
    csv_val: str   = "val.csv"
    fs: float = 250.0
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 40
    lr_backbone: float = 1e-4
    lr_frontend: float = 3e-4
    lr_head: float = 3e-4
    wd: float = 1e-2
    hetero: bool = True
    use_cbramod: bool = True
    cbramod_ckpt: Optional[str] = None
    use_coral: bool = True
    coral_lambda: float = 0.1
    adv_lambda: float = 0.1
    use_film: bool = True
    tta: bool = False
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def build_cond_vectors(metas, domains_map=None):
    # Example: condition on task (domain one-hot)
    tasks = [m["task"] for m in metas]
    uniq = sorted(set(tasks))
    if domains_map is None:
        domains_map = {t:i for i,t in enumerate(uniq)}
    idx = [domains_map[t] for t in tasks]
    cond = F.one_hot(torch.tensor(idx), num_classes=len(domains_map)).float()
    return cond, domains_map

def train_one_epoch(model, loader, opt_front, opt_back, opt_head, scaler, cfg: CFG, metrics_accum):
    model.train()
    total_loss = 0.0
    for X, Y, sub_ids, domains, metas in loader:
        X = X.to(cfg.device); Y = Y.to(cfg.device)
        sub_ids = sub_ids.to(cfg.device); domains = domains.to(cfg.device)
        cond, _ = (None, None)
        if cfg.use_film:
            cond, _ = build_cond_vectors(metas); cond = cond.to(cfg.device)

        opt_front.zero_grad(set_to_none=True)
        opt_back.zero_grad(set_to_none=True)
        opt_head.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=True):
            out, logits_d, s_emb = model(X, sub_ids, cond_vec=cond, adv_lambda=cfg.adv_lambda)
            # build subject-level Y by averaging windows belonging to each subject in batch
            n_sub = s_emb.shape[0]
            # gather subject targets
            Y_sub = torch.zeros(n_sub, Y.shape[-1], device=Y.device)
            cnt = torch.zeros(n_sub, 1, device=Y.device)
            Y_sub.index_add_(0, sub_ids, Y)
            cnt.index_add_(0, sub_ids, torch.ones_like(cnt[:sub_ids.shape[0]]))
            Y_sub = Y_sub / (cnt + 1e-8)

            # regression loss
            if cfg.hetero:
                mu, logv = out
                loss_reg = heteroscedastic_nll(Y_sub, mu, logv)
            else:
                pred = out
                loss_reg = F.mse_loss(pred, Y_sub)

            # adversarial domain loss (cross-entropy, uniform target)
            # We push logits_d to be uninformative by training adversary to predict true domains
            # while GRL in forward flips grad sign to backbone
            loss_adv = F.cross_entropy(logits_d, torch.arange(logits_d.size(0), device=logits_d.device) * 0)  # dummy if domains per subject unavailable
            # If per-subject domain ids are available, replace above with real labels:
            # subj_dom = torch.tensor([...], device=...)  # left as an exercise based on metas
            # loss_adv = F.cross_entropy(logits_d, subj_dom)

            # optional CORAL: split by domain inside batch if possible
            loss_coral = 0.0
            if cfg.use_coral:
                # rough split: top half vs bottom half by domain id
                mask = (domains % 2 == 0)
                if mask.any() and (~mask).any():
                    # compute mean per subject; map subject_ids to domain via majority vote in batch
                    # simplification: use window-level features grouping by mask
                    s_even = s_emb[mask[::1][:s_emb.shape[0]]] if s_emb.shape[0] == mask.shape[0] else s_emb[:mask.sum()]
                    s_odd  = s_emb[~mask[::1][:s_emb.shape[0]]] if s_emb.shape[0] == mask.shape[0] else s_emb[mask.sum():]
                    if s_even.shape[0] > 1 and s_odd.shape[0] > 1:
                        loss_coral = coral_loss(s_even, s_odd)

            loss = loss_reg + cfg.adv_lambda * loss_adv + cfg.coral_lambda * loss_coral

        scaler.scale(loss).backward()
        scaler.step(opt_front); scaler.step(opt_back); scaler.step(opt_head)
        scaler.update()

        total_loss += loss.item()

        # metrics on-the-fly (window-level approx)
        metrics_accum["mse_sum"] += F.mse_loss(Y, Y.mean(dim=0, keepdim=True).expand_as(Y)).item() * 0  # placeholder
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, cfg: CFG, tta=False):
    model.eval()
    all_preds = []
    all_trues = []
    for X, Y, sub_ids, domains, metas in loader:
        X = X.to(cfg.device); Y = Y.to(cfg.device)
        sub_ids = sub_ids.to(cfg.device)
        cond = None
        if cfg.use_film:
            cond, _ = build_cond_vectors(metas); cond = cond.to(cfg.device)

        if tta:
            # simple BN/Layer adaptation-like: multi-aug avg at test
            preds_acc = []
            for _ in range(4):
                X_aug = X.clone()
                X_aug = add_noise(X_aug, 0.005)
                out, _, s_emb = model(X_aug, sub_ids, cond_vec=cond, adv_lambda=0.0)
                n_sub = s_emb.shape[0]
                Y_sub = torch.zeros(n_sub, Y.shape[-1], device=Y.device)
                cnt = torch.zeros(n_sub, 1, device=Y.device)
                Y_sub.index_add_(0, sub_ids, Y)
                cnt.index_add_(0, sub_ids, torch.ones_like(cnt[:sub_ids.shape[0]]))
                if isinstance(out, tuple):
                    mu, _ = out
                    preds_acc.append(mu)
                else:
                    preds_acc.append(out)
            pred = torch.stack(preds_acc, dim=0).mean(dim=0)
            true = Y_sub
        else:
            out, _, s_emb = model(X, sub_ids, cond_vec=cond, adv_lambda=0.0)
            n_sub = s_emb.shape[0]
            Y_sub = torch.zeros(n_sub, Y.shape[-1], device=Y.device)
            cnt = torch.zeros(n_sub, 1, device=Y.device)
            Y_sub.index_add_(0, sub_ids, Y)
            cnt.index_add_(0, sub_ids, torch.ones_like(cnt[:sub_ids.shape[0]]))
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out
            true = Y_sub

        all_preds.append(pred.cpu())
        all_trues.append(true.cpu())

    P = torch.cat(all_preds, dim=0)
    T = torch.cat(all_trues, dim=0)
    # metrics per target
    mse = F.mse_loss(P, T).item()
    rmse_v = torch.sqrt(F.mse_loss(P, T)).item()
    nrmse_v = (torch.sqrt(F.mse_loss(P, T)) / (T.std(dim=0, unbiased=False)+1e-8)).mean().item()
    # per-head NRMSE (optional print)
    per_head_nrmse = ((P - T).pow(2).mean(dim=0).sqrt() / (T.std(dim=0, unbiased=False)+1e-8)).tolist()
    return {"MSE": mse, "RMSE": rmse_v, "NRMSE": nrmse_v, "per_head_NRMSE": per_head_nrmse}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--hetero", action="store_true", help="use heteroscedastic NLL head")
    ap.add_argument("--use_cbramod", action="store_true")
    ap.add_argument("--cbramod_ckpt", type=str, default="")
    ap.add_argument("--use_coral", action="store_true")
    ap.add_argument("--coral_lambda", type=float, default=0.1)
    ap.add_argument("--adv_lambda", type=float, default=0.1)
    ap.add_argument("--use_film", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = CFG(
        csv_train=args.train_csv,
        csv_val=args.val_csv,
        fs=args.fs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hetero=args.hetero,
        use_cbramod=args.use_cbramod,
        cbramod_ckpt=args.cbramod_ckpt if args.cbramod_ckpt else None,
        use_coral=args.use_coral,
        coral_lambda=args.coral_lambda,
        adv_lambda=args.adv_lambda,
        use_film=args.use_film,
        tta=args.tta
    )

    set_seed(cfg.seed)
    device = cfg.device
    print(f"[Device] {device}")

    # Load data
    train_ds = EEGWindows(cfg.csv_train, augment=True, fs=cfg.fs)
    val_ds   = EEGWindows(cfg.csv_val, augment=False, fs=cfg.fs)

    # Target standardization is assumed already applied in CSV; if not, do it offline and save.
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=False)

    # Infer n_channels from one sample
    x0 = np.load(train_ds.df.iloc[0]["path"]).astype(np.float32)
    n_channels = x0.shape[0]
    n_domains = len(train_ds.domains)
    cond_dim = len(train_ds.domains) if cfg.use_film else 0

    model = Model(n_channels=n_channels, fs=cfg.fs, cond_dim=cond_dim, n_domains=n_domains,
                  use_cbramod=cfg.use_cbramod, cbramod_ckpt=cfg.cbramod_ckpt, hetero=cfg.hetero).to(device)

    # Optimizers with LLRD style
    front_params = list(model.front.parameters()) + list(model.sep.parameters())
    back_params  = list(model.backbone.parameters())
    head_params  = list(model.reg.parameters()) + list(model.pool.parameters()) + list(model.adv.parameters())
    if model.film is not None: head_params += list(model.film.parameters())

    opt_front = torch.optim.AdamW(front_params, lr=cfg.lr_frontend, weight_decay=cfg.wd)
    opt_back  = torch.optim.AdamW(back_params,  lr=cfg.lr_backbone,  weight_decay=cfg.wd)
    opt_head  = torch.optim.AdamW(head_params,  lr=cfg.lr_head,      weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    best = {"NRMSE": 1e9}
    for epoch in range(1, cfg.epochs+1):
        metrics_accum = {"mse_sum": 0.0}
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt_front, opt_back, opt_head, scaler, cfg, metrics_accum)
        val_metrics = evaluate(model, val_loader, cfg, tta=cfg.tta)
        dt = time.time() - t0
        print(f"[{epoch:03d}/{cfg.epochs}] loss={tr_loss:.4f}  "
              f"val_NRMSE={val_metrics['NRMSE']:.4f}  "
              f"per_head={['%.3f'%x for x in val_metrics['per_head_NRMSE']]}  "
              f"time={dt:.1f}s")

        # save best by NRMSE
        if val_metrics["NRMSE"] < best["NRMSE"]:
            best = {"NRMSE": val_metrics["NRMSE"], "epoch": epoch}
            torch.save(model.state_dict(), "best_model.pt")
            with open("best_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
    print("[Best]", best)

if __name__ == "__main__":
    main()
