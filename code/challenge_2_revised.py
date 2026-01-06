"""
challenge2_with_cache.py

Usage:
# 1) 캐시 생성 (한 번만 실행)
python challenge2_with_cache.py --build-cache

# 2) 학습 실행
python challenge2_with_cache.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import gc
from einops import rearrange
from einops.layers.torch import Rearrange

# ---------------------------
# 설정 (사용자에 맞게 수정)
# ---------------------------
PREPROCESSED_ROOT = "/data5/open_data/HBN/Preprocessed_EEG/0922try_bySubject/"
CACHE_ROOT = "/data5/open_data/HBN/Preprocessed_EEG/cache_segments/"  # 캐시 저장 경로
BIDS_ROOT = "/data5/open_data/HBN/EEG_BIDS"
CBRAMOD_PATH = "/home/RA/EEG_Challenge/Challenge2/CBraMod"
PRETRAINED_MODEL_PATH = "/home/RA/EEG_Challenge/Challenge2/CBraMod/pretrained_weights/pretrained_weights.pth"

if CBRAMOD_PATH not in sys.path:
    sys.path.append(CBRAMOD_PATH)
from models.cbramod import CBraMod  # 로컬환경에서만 작동. 제출 시에는 클래스 포함 필요.

N_CHANNELS = 128
N_TIMES = 400
PATCH_SIZE = 200
SEQ_LEN = N_TIMES // PATCH_SIZE

BATCH_SIZE = 64
FINETUNE_EPOCHS = 5
LR = 1e-4  # 기본 LR 유지
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_OUTPUTS = 4

# ---------------------------
# 유틸: participants 로드
# ---------------------------
def load_all_participants(bids_root, datasets):
    required_cols = ['participant_id', 'p_factor', 'internalizing', 'externalizing', 'attention']
    all_dfs = []
    for ds in datasets:
        pfile = os.path.join(bids_root, ds, "participants.tsv")
        if not os.path.exists(pfile):
            continue
        try:
            df = pd.read_csv(pfile, sep="\t")
            # 일부 파일 컬럼 이름이 다를 수 있으므로 안전하게 재인덱싱
            if 'participant_id' not in df.columns:
                continue
            # keep columns if exist
            cols = [c for c in required_cols if c in df.columns]
            df_sel = df[['participant_id'] + [c for c in cols if c!='participant_id']]
            all_dfs.append(df_sel)
        except Exception:
            continue
    if not all_dfs:
        return pd.DataFrame(columns=required_cols)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined['participant_id'] = combined['participant_id'].str.strip()
    # keep rows with any non-null of target columns:
    return combined.drop_duplicates(subset=['participant_id']).reset_index(drop=True)

# ---------------------------
# 캐시 빌드 (한 번만 실행)
# ---------------------------
def build_cache(preprocessed_root, cache_root, force=False):
    """
    각 subject의 전처리된 .set 파일을 찾아 .npy로 저장.
    - 파일명 예시: sub-XXXX_task-*_eeg_pp.set
    - 캐시 파일: cache_root/sub-XXXX.npy (shape: (n_chan, n_times_total))
    """
    os.makedirs(cache_root, exist_ok=True)
    files = glob(os.path.join(preprocessed_root, "sub-*", "*_eeg_pp.set"), recursive=True)
    print(f"Found {len(files)} preprocessed .set files.")

    for f in tqdm(files, desc="Building cache"):
        try:
            basename = os.path.basename(f)
            subj = basename.split("sub-")[1].split("_")[0]
            cache_file = os.path.join(cache_root, f"sub-{subj}.npy")
            if os.path.exists(cache_file) and not force:
                continue
            # 안전하게 로드: preload=True로 로드한 뒤 저장하고 삭제
            raw = mne.io.read_raw_eeglab(f, preload=True, verbose=False)
            data = raw.get_data()  # (n_chan, n_times)
            # ensure channel count matches expected (skip otherwise)
            if data.shape[0] != N_CHANNELS:
                # optionally we could interpolate here, but skip for cache simplicity
                raw.close()
                continue
            # save as float32
            np.save(cache_file, data.astype(np.float32), allow_pickle=False)
            raw.close()
            # free memory
            del raw, data
            gc.collect()
        except Exception:
            # skip problematic files
            continue
    print("Cache build complete.")

# ---------------------------
# Cached Dataset (mmap 사용)
# ---------------------------
class CachedEEGDataset(Dataset):
    """
    캐시된 subject-level .npy 파일을 메모리맵(mmap_mode='r')으로 열어 segment 단위로 제공.
    sample 형식: (C, segment_len)
    """
    def __init__(self, cache_root, file_label_pairs, labels_df, segment_len=N_TIMES, segments_per_subject=None, training=True):
        """
        - cache_root: 캐시된 .npy 파일들이 있는 폴더
        - file_label_pairs: [(orig_set_path, "") ...] (원본 파일 리스트에서 추출한 튜플, subject id 파싱에 사용)
        - labels_df: participants dataframe (participant_id 컬럼 존재)
        - segments_per_subject: None (use all) or int (subsampling per subject)
        """
        self.cache_root = cache_root
        self.segment_len = segment_len
        self.labels_df = labels_df.set_index("participant_id")
        self.training = training
        self.index = []  # list of (cache_path, start_idx, subj_id)

        # build index by scanning files or cache files
        # file_label_pairs used to get subj list in same order as previous pipeline
        for f, _ in file_label_pairs:
            try:
                basename = os.path.basename(f)
                subj = basename.split("sub-")[1].split("_")[0]
                subj_id = f"sub-{subj}"
                if subj_id not in self.labels_df.index:
                    continue
                cache_file = os.path.join(cache_root, f"{subj_id}.npy")
                if not os.path.exists(cache_file):
                    continue
                # open memmap to get shape without loading
                arr = np.load(cache_file, mmap_mode='r')
                n_times = arr.shape[1]
                n_segments = n_times // segment_len
                if n_segments <= 0:
                    continue
                # default: index all segments per subject
                for i in range(n_segments):
                    self.index.append((cache_file, i * segment_len, subj_id))
                arr = None
            except Exception:
                continue

        # optional shuffle at dataset init for variety (not same as DataLoader shuffle)
        # but we keep it deterministic here
        # if segments_per_subject is set, we subsample per subject
        if segments_per_subject is not None:
            # select at most segments_per_subject per subject
            grouped = {}
            for cache_file, start, subj_id in self.index:
                grouped.setdefault(subj_id, []).append((cache_file, start))
            new_index = []
            for subj_id, items in grouped.items():
                if len(items) <= segments_per_subject:
                    new_index.extend(items)
                else:
                    sel = np.random.choice(len(items), size=segments_per_subject, replace=False)
                    new_index.extend([items[s] for s in sel])
            self.index = new_index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cache_file, start, subj = self.index[idx]
        # open memmap (fast, thread-safe)
        arr = np.load(cache_file, mmap_mode='r')  # shape (C, T)
        # 중요: .copy()를 추가하여 읽기 전용 뷰가 아닌, 쓰기 가능한 배열로 복사
        seg = arr[:, start:start+self.segment_len].copy()
        # if short, pad (shouldn't happen if indexed properly)
        if seg.shape[1] != self.segment_len:
            pad = np.zeros((arr.shape[0], self.segment_len - seg.shape[1]), dtype=np.float32)
            seg = np.concatenate([seg, pad], axis=1)

        # -----------------------------
        # 🌟 NaN 방지: 세그먼트 단위 표준화 (Normalization) 추가 
        # -----------------------------
        # EEG 데이터의 큰 스케일로 인한 NaN 방지를 위해 필수
        seg_mean = seg.mean(axis=1, keepdims=True)
        seg_std = seg.std(axis=1, keepdims=True)
        # 분모가 0이 되는 것을 방지하기 위해 작은 값(1e-8)을 더해줍니다.
        seg = (seg - seg_mean) / (seg_std + 1e-8)
        # -----------------------------

        # optional augmentation (only training)
        if self.training:
            # simple channel dropout example
            if np.random.rand() < 0.5:
                drop_n = int(N_CHANNELS * 0.05)
                drop_idx = np.random.choice(N_CHANNELS, drop_n, replace=False)
                seg[drop_idx, :] = 0.0
            # time mask
            if np.random.rand() < 0.5:
                mask_len = int(self.segment_len * 0.1)
                if mask_len > 0:
                    s = np.random.randint(0, max(1, self.segment_len - mask_len))
                    seg[:, s:s+mask_len] = 0.0

        # label
        label = self.labels_df.loc[subj].values.astype(np.float32)
        x = torch.from_numpy(seg).float()  # (C, segment_len)
        y = torch.from_numpy(label).float()
        return x, y

# ---------------------------
# 모델 (간단히 기존 사용한 구조 재사용)
# ---------------------------
class EEGRegressor(nn.Module):
    def __init__(self, encoder, n_outputs=N_OUTPUTS):
        super().__init__()
        self.encoder = encoder
        self.encoder.proj_out = nn.Identity()
        with torch.no_grad():
            dummy = torch.randn(1, N_CHANNELS, SEQ_LEN, PATCH_SIZE)
            out = self.encoder(dummy)
            flat = out.view(1, -1).shape[1]
        self.head = nn.Sequential(
            Rearrange('b c s p -> b (c s p)'),
            nn.Linear(flat, 512),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_outputs)
        )
    def forward(self, x):
        # x: (B, C, T) -> (B, C, S, P)
        x = rearrange(x, 'b c (s p) -> b c s p', s=SEQ_LEN, p=PATCH_SIZE)
        feat = self.encoder(x)
        return self.head(feat)

# ---------------------------
# 학습 루프 (AMP 사용)
# ---------------------------
def finetune(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    
    # 🌟 NaN 방지: Encoder와 Head에 차등 LR 적용 (Encoder는 더 낮게)
    optimizer_grouped_parameters = [
        {'params': model.encoder.parameters(), 'lr': lr * 0.1}, # 1e-5 (Encoder)
        {'params': model.head.parameters(), 'lr': lr}            # 1e-4 (Head)
    ]
    opt = torch.optim.Adam(optimizer_grouped_parameters)
    
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.MSELoss()
    best = float('inf')

    for ep in range(epochs):
        model.train()
        running = 0.0
        for X, y in tqdm(train_loader, desc=f"Train Ep{ep+1}"):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                preds = model(X)
                loss = criterion(preds, y)
            scaler.scale(loss).backward()
            # 🌟 NaN 방지: Gradients를 0으로 클리핑하는 대신, AMP Scaler를 사용
            # 클리핑 없이도 데이터 표준화와 차등 LR로 충분히 안정화 가능
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        avg_train = running / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                    preds = model(X)
                    val_loss += criterion(preds, y).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {ep+1}/{epochs} Train {avg_train:.4f} Val {avg_val:.4f}")
        # save best
        if avg_val < best:
            best = avg_val
            torch.save(model.state_dict(), "best_finetuned_cached.pth")
            print("Saved best model.")
        # free
        torch.cuda.empty_cache()
        gc.collect()
    return model

# ---------------------------
# main: 캐시 빌드 또는 학습 실행
# ---------------------------
def main(args):
    # datasets lists
    train_datasets = [f"ds00{i}" for i in range(5505,5517) if i != 5509]
    val_datasets = ["ds005509"]
    df_train = load_all_participants(BIDS_ROOT, train_datasets)
    df_val = load_all_participants(BIDS_ROOT, val_datasets)

    # collect original .set files (to preserve ordering / matching)
    all_pp = glob(os.path.join(PREPROCESSED_ROOT, "sub-*", "*_eeg_pp.set"), recursive=True)
    all_pp = [(f, "") for f in all_pp]

    if args.build_cache:
        print("Building cache (this may take time).")
        build_cache(PREPROCESSED_ROOT, CACHE_ROOT, force=args.force)
        print("Cache built. Exit.")
        return

    # build train/val file lists by participant membership
    train_ids = set(df_train['participant_id'])
    val_ids = set(df_val['participant_id'])
    train_files, val_files = [], []
    for f, e in all_pp:
        try:
            subj = "sub-" + os.path.basename(f).split("sub-")[1].split("_")[0]
            if subj in train_ids:
                train_files.append((f,e))
            elif subj in val_ids:
                val_files.append((f,e))
        except Exception:
            continue

    # Datasets (cached)
    # segments_per_subject=None 은 모든 세그먼트 사용
    train_dataset = CachedEEGDataset(CACHE_ROOT, train_files, df_train, segment_len=N_TIMES, training=True)
    val_dataset = CachedEEGDataset(CACHE_ROOT, val_files, df_val, segment_len=N_TIMES, training=False)

    # DataLoaders: we can use multiple workers because IO is mmap (safe)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # model
    encoder = CBraMod(in_dim=PATCH_SIZE, seq_len=SEQ_LEN)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        sd = torch.load(PRETRAINED_MODEL_PATH, map_location='cpu')
        encoder.load_state_dict(sd, strict=False)
        print("Loaded pretrained encoder (partial).")
    else:
        print("No pretrained encoder found - training from scratch.")

    model = EEGRegressor(encoder, n_outputs=N_OUTPUTS)
    model = finetune(model, train_loader, val_loader, FINETUNE_EPOCHS, LR, DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cache", action="store_true", help="Build cached .npy files from .set files")
    parser.add_argument("--force", action="store_true", help="Force rebuild cache even if exists")
    args = parser.parse_args()
    main(args)