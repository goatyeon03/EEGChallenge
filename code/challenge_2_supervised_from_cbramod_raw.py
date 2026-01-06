import os, random, time, warnings, json, re, glob
import numpy as np, pandas as pd, torch, mne
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
BIDS_ROOT   = "/data5/open_data/HBN/EEG_BIDS"
CACHE_DIR = "/data5/open_data/HBN/Preprocessed_EEG/0922try_bySubject"
# MISSING_TXT = "/home/RA/EEG_Challenge/Challenge2/logs/missing_cache_files.txt"
CBRAMOD_WEIGHTS = "/home/RA/EEG_Challenge/Challenge2/CBraMod/pretrained_weights/pretrained_weights.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SFREQ  = 100
CROP_T = 1000
N_CHANS = 128   # Cz 제거 후
BATCH   = 8
EPOCHS  = 10
LR_FRONT, LR_BACKB, LR_HEAD = 3e-4, 1e-4, 3e-4
WD = 1e-2
SEED = 42

VAL_RELEASE = "R5"

VALID_SUBJECT_CSV = "/home/RA/EEG_Challenge/Challenge2/valid_subjects_with_release.csv"

warnings.filterwarnings("ignore")
mne.set_log_level("CRITICAL")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ============================================================
# 1. 파일 리스트 구성
# ============================================================
def build_raw_filelist(bids_root, subjects_to_find):
    """
    BIDS_ROOT를 스캔하여 subjects_to_find 목록에 있는
    대상자들의 모든 .set 파일 경로를 찾습니다.
    (subjects_to_find는 set이나 list 형태)
    """
    eeg_files = []
    subject_set = set(subjects_to_find)
    print(f"Building raw .set file list for {len(subject_set)} subjects...")
    
    # BIDS_ROOT/ds*/sub-*/eeg/*.set 패턴 검색
    search_pattern = os.path.join(bids_root, "ds*", "sub-*", "eeg", "*.set")
    all_set_files = glob.glob(search_pattern)
    
    if not all_set_files:
        print(f"⚠️  경고: '{search_pattern}' 경로에서 .set 파일을 찾을 수 없습니다.")
        return []

    # 파일 경로에서 'sub-XXXX' 추출
    subject_pattern = re.compile(r'(sub-[a-zA-Z0-9]+)')

    for f_path in tqdm(all_set_files, desc="Scanning .set files"):
        match = subject_pattern.search(f_path)
        if match:
            subj_id = match.group(1)
            if subj_id in subject_set:
                eeg_files.append(f_path)
                
    print(f"✅ Found {len(eeg_files)} raw .set files for the valid subjects.")
    return sorted(list(set(eeg_files)))


# ============================================================
# 2. 메타데이터 로드
# ============================================================
def load_participants(bids_root):
    dfs=[]
    for ds in sorted(os.listdir(bids_root)):
        pfile=os.path.join(bids_root, ds, "participants.tsv")
        if os.path.exists(pfile):
            df=pd.read_csv(pfile,sep="\t")
            df["release_number"]=ds ##########
            dfs.append(df)
    meta=pd.concat(dfs,ignore_index=True)
    meta.columns=[c.lower() for c in meta.columns]
    meta=meta.drop_duplicates(subset=["participant_id"]).set_index("participant_id")

    # p_factor, age, sex, ehq_total 정리
    if "p_factor" not in meta.columns:
        alt=[c for c in meta.columns if "p" in c and "factor" in c]
        if alt: meta=meta.rename(columns={alt[0]:"p_factor"})
    meta["sex"]=meta["sex"].map({"M":1,"F":0,"m":1,"f":0}).fillna(0.5)
    for c in ["age","ehq_total"]:
        if c not in meta.columns: meta[c]=np.nan
        meta[c]=meta[c].fillna(meta[c].mean())
        m,s=meta[c].mean(),meta[c].std()+1e-8
        meta[c]=(meta[c]-m)/s
    return meta

# ============================================================
# 3. Dataset
# ============================================================
# ============================================================
# 3. Dataset
# ============================================================
def subj_from_fname(fname):
    base=os.path.basename(fname)
    return base.split("_")[0]

class RawEEGDataset(Dataset):
    def __init__(self, files, meta, train=True):
        self.files=files; self.meta=meta; self.train=train
        self.targets=meta["p_factor"].dropna()
        self.subj_ids=set(self.targets.index)

    def __len__(self): return len(self.files)

    def __getitem__(self,i):
        path=self.files[i]
        subj=subj_from_fname(path)

        # ⭐️ [수정] 실패 시 반환할 더미 데이터를 만드는 함수
        def get_dummy():
            return torch.zeros((N_CHANS, CROP_T), dtype=torch.float32), \
                   torch.tensor(0., dtype=torch.float32), \
                   torch.zeros(3, dtype=torch.float32)

        if subj not in self.subj_ids:
            return get_dummy() # 대상자 ID가 없으면 더미 반환

        # ⭐️ [수정] 파일 로딩 및 처리 전체를 try-except로 감쌉니다.
        try:
            raw=mne.io.read_raw_eeglab(path,preload=True,verbose=False)
            raw.filter(0.5,50.,fir_design="firwin")
            raw.set_eeg_reference("average",projection=False)
            raw.resample(SFREQ)
            x=raw.get_data()
            
            if x.shape[0]==129: 
                x=x[:-1,:]  # Cz 제거
            
            # ⭐️ [수정] 채널 수가 N_CHANS (128)와 일치하는지 명시적으로 확인
            if x.shape[0] != N_CHANS:
                # print(f"Warning: Skipping {path}, expected {N_CHANS} channels, got {x.shape[0]}")
                return get_dummy() # 채널 수가 안 맞으면 더미 반환

            x=(x-x.mean(axis=1,keepdims=True))/(x.std(axis=1,keepdims=True)+1e-6)

            T=x.shape[1]
            if T<CROP_T:
                pad=np.zeros((x.shape[0],CROP_T)); pad[:,:T]=x; x=pad
            else:
                st=np.random.randint(0,T-CROP_T+1) if self.train else (T-CROP_T)//2
                x=x[:,st:st+CROP_T]

            # ⭐️ [수정] 최종 shape 확인 (혹시 모를 오류 방지)
            if x.shape != (N_CHANS, CROP_T):
                # print(f"Warning: Skipping {path}, final shape mismatch {x.shape}")
                return get_dummy()

            y=float(self.meta.loc[subj,"p_factor"]) if subj in self.meta.index else 0.
            meta_vec=torch.tensor([
                self.meta.loc[subj,"age"],
                self.meta.loc[subj,"sex"],
                self.meta.loc[subj,"ehq_total"]
            ],dtype=torch.float32) if subj in self.meta.index else torch.zeros(3)
            
            return torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32), meta_vec
        
        except Exception as e:
            # ⭐️ [수정] MNE 로딩 실패 등 다른 모든 예외 발생 시
            # print(f"Error processing file {path}: {e}")
            return get_dummy() # 문제가 있는 파일은 더미 데이터 반환

# ============================================================
# 4. 모델 정의 (CBraMod + regression head)
# ============================================================
class SincConv1d(nn.Module):
    def __init__(self, out_channels=64, kernel_size=129, sample_rate=100, min_hz=0.3, max_hz=45.0):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.sample_rate  = sample_rate
        self.min_hz       = float(min_hz)
        self.max_hz       = float(max_hz)
        low  = torch.linspace(self.min_hz, self.max_hz - 5.0, out_channels)
        band = torch.ones(out_channels) * 5.0
        self.low_hz_  = nn.Parameter(low)
        self.band_hz_ = nn.Parameter(band)
        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        self.register_buffer("n", n)

    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        device  = x.device
        dtype   = x.dtype

        # low: (out,)
        low  = torch.clamp(torch.abs(self.low_hz_), min=self.min_hz, max=self.max_hz - 1.0)
        # high: clamp with tensor min/max (same shape)
        raw_high = low + torch.abs(self.band_hz_)
        min_v    = low + 1.0
        max_v    = torch.full_like(low, self.max_hz)
        high     = torch.clamp(raw_high, min=min_v, max=max_v)

        n = self.n.to(device=device, dtype=dtype)
        window = torch.hamming_window(self.kernel_size, periodic=False, dtype=dtype, device=device)

        filters = []
        nyq = (self.sample_rate / 2.0)
        for i in range(self.out_channels):
            f1 = low[i]  / nyq
            f2 = high[i] / nyq
            h1 = 2 * f2 * torch.sinc(2 * f2 * n)
            h2 = 2 * f1 * torch.sinc(2 * f1 * n)
            bandpass = (h1 - h2) * window
            filters.append(bandpass)
        filt = torch.stack(filters, dim=0).unsqueeze(1)  # (out, 1, K)

        x_dw = x.view(B * C, 1, T) # (B*C, 1, T)
        
        # [수정] x 대신 x_dw 를 입력으로 사용합니다.
        y = F.conv1d(x_dw, filt, stride=1, padding=self.kernel_size // 2) # ✅ 수정된 라인

        # 이제 y의 shape은 [1024, 64, 1000] 이 됩니다.
        
        y = y.view(B, C, self.out_channels, y.shape[-1]).sum(dim=1)  # (B, out, T)
        return y


class SEBlock(nn.Module):
    def __init__(self,c,r=8):
        super().__init__()
        self.fc1=nn.Linear(c,c//r); self.fc2=nn.Linear(c//r,c)
    def forward(self,x):
        s=x.mean(-1); e=F.relu(self.fc1(s)); e=torch.sigmoid(self.fc2(e)).unsqueeze(-1)
        return x*e

class CBraModBackbone(nn.Module):
    def __init__(self,out_dim=512):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv1d(64,128,7,padding=3),nn.ReLU(),
            nn.Conv1d(128,256,5,padding=2),nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc=nn.Linear(256,out_dim); self.out_dim=out_dim
    def load_weights(self,wpath):
        sd=torch.load(wpath,map_location="cpu")
        self.load_state_dict(sd,strict=False)
        print(f"[Info] Loaded pretrained CBraMod weights: {wpath}")
    def forward(self,x):
        h=self.conv(x).squeeze(-1)
        return self.fc(h)

class RegressionHead(nn.Module):
    def __init__(self,in_dim,meta_dim=3,hetero=True):
        super().__init__()
        self.hetero=hetero
        self.meta_fc=nn.Sequential(nn.Linear(meta_dim,32),nn.ReLU())
        self.fc=nn.Sequential(nn.Linear(in_dim+32,128),nn.ReLU(),nn.Linear(128,1))
        if hetero:
            self.logv=nn.Sequential(nn.Linear(in_dim+32,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,f,meta):
        
        m=self.meta_fc(meta)
        z=torch.cat([f,m],1)
        
        mu=self.fc(z).squeeze(-1)
        if self.hetero:
            lv = self.logv(z)
            
            if isinstance(lv, (tuple, list)):
                lv = lv[0]
            logv = lv.squeeze(-1)
            logv = torch.clamp(logv, min=-6.0, max=6.0)

            return mu,logv

def hetero_nll(y,mu,logv):
    inv=torch.exp(-logv)
    return 0.5*((y-mu)**2*inv+logv).mean()

# ============================================================
# 5. 학습 루프
# ============================================================
def train_epoch(model,head,loader,opt,scaler,hetero):
    model.train(); head.train(); total=0
    for x,y,meta in tqdm(loader,desc="Train",leave=False):
        x,y,meta=x.to(DEVICE),y.to(DEVICE),meta.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(True):
            z=model(x); 
            out=head(z,meta)
            if hetero:
                mu,logv=out; loss=hetero_nll(y,mu,logv)
            else:
                mu=out; loss=F.mse_loss(mu,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        total+=loss.item()
    return total/len(loader)

@torch.no_grad()
def evaluate(model,head,loader,hetero):
    model.eval(); head.eval()
    preds,trues=[],[]
    for x,y,meta in tqdm(loader,desc="Val",leave=False):
        x,meta=x.to(DEVICE),meta.to(DEVICE)
        out=head(model(x),meta)
        mu=out[0] if hetero else out
        preds.extend(mu.detach().cpu().numpy()); trues.extend(y.numpy())
    preds,trues=np.array(preds),np.array(trues)
    mae=np.mean(np.abs(preds-trues))
    mse=np.mean((preds-trues)**2)
    r2=1-np.sum((preds-trues)**2)/np.sum((trues-trues.mean())**2)
    nrmse=np.sqrt(mse)/(trues.std()+1e-8)
    return mae,mse,r2,nrmse

# ============================================================
# 6. 실행
# ============================================================
def main():
    # 1. 모든 참가자 메타데이터 로드
    meta = load_participants(BIDS_ROOT) # 'release_number' ('ds...') 포함

    # 2. 챌린지 2용 유효 대상자 목록(CSV) 로드
    if not os.path.exists(VALID_SUBJECT_CSV):
        raise FileNotFoundError(f"유효 대상자 CSV 파일을 찾을 수 없습니다: {VALID_SUBJECT_CSV}")
    
    valid_df = pd.read_csv(VALID_SUBJECT_CSV)
    valid_subject_ids = set(valid_df['participant_id'])
    print(f"Loaded {len(valid_subject_ids)} valid subject IDs from {VALID_SUBJECT_CSV}.")

    # --- [수정 시작] ---
    
    # 3. CSV의 'release_number' 컬럼명을 'release'로 변경 (★ 충돌 회피 ★)
    if 'release_number' in valid_df.columns:
        valid_df = valid_df.rename(columns={'release_number': 'release'})
    elif 'release' not in valid_df.columns:
        raise ValueError(f"CSV 파일에 'release' 또는 'release_number' 컬럼이 없습니다.")
    
    # 'release' 컬럼('R5' 등)을 'meta' DataFrame에 병합
    valid_df_subset = valid_df[['participant_id', 'release']].set_index('participant_id')
    
    # meta(index, 'release_number') + valid_df_subset(index, 'release')
    meta = meta.join(valid_df_subset) # 이제 충돌 없음

    # 4. 메타데이터 필터링
    meta = meta.reindex(valid_subject_ids) # CSV 기준으로 인덱스 재설정
    meta = meta[meta['p_factor'].notna()]  # P-factor 없는 대상자 제거
    
    # (디버깅) 'release' 컬럼이 제대로 병합되었는지 확인 (★ 수정 ★)
    if VAL_RELEASE in meta['release'].values:
        print(f"✅ Successfully joined 'release' column (found '{VAL_RELEASE}').")
    else:
        print(f"⚠️  WARNING: Could not find '{VAL_RELEASE}' in 'release' column after join.")
        
    print(f"Filtered to {len(meta.index)} subjects with valid demo + p_factor.")

    # 5. Train/Val 대상자 ID 분리 (★ 수정 ★)
    #    'release' (R5) 또는 'release_number' (ds005510) 확인
    val_mask = (meta['release'] == VAL_RELEASE) | (meta['release_number'] == 'ds005510')
    val_subjects = set(meta[val_mask].index)
    train_subjects = set(meta.index) - val_subjects
    
    # --- [수정 끝] ---
    
    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")

    # 6. 대상자에 해당하는 .set 파일 리스트 생성 (기존: 5단계)
    all_files = build_raw_filelist(BIDS_ROOT, meta.index) # meta.index = train + val

    # 7. 파일을 Train/Val로 분리 (기존 코드와 동일)
    train_files = [f for f in all_files if subj_from_fname(f) in train_subjects]
    val_files = [f for f in all_files if subj_from_fname(f) in val_subjects]
    
    if len(val_subjects) == 0 or len(val_files) == 0:
        print("\n" + "="*50)
        print("❌ CRITICAL ERROR: Validation set is empty.")
        print(" 'release' 또는 'release_number' 컬럼 매칭에 실패했습니다.")
        print(" 'meta' DataFrame의 컬럼을 확인하세요:")
        # (★ 수정 ★) 충돌 해결된 두 컬럼을 모두 출력
        if 'release' in meta.columns:
            print(meta[['release', 'release_number']].value_counts())
        else:
            print(meta[['release_number']].value_counts())
        print("="*50 + "\n")
        raise ValueError("Validation set construction failed.")
        
    print(f"Total files: {len(all_files)} (Train: {len(train_files)}, Val: {len(val_files)})")

    # 7. Dataset 및 DataLoader 생성 (RawEEGDataset 사용)
    train_ds = RawEEGDataset(train_files, meta, train=True)
    val_ds   = RawEEGDataset(val_files, meta, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    # Model
    front=SincConv1d(out_channels=64,kernel_size=129,sample_rate=SFREQ)
    se=SEBlock(64); backbone=CBraModBackbone(out_dim=512)
    backbone.load_weights(CBRAMOD_WEIGHTS)
    model=nn.Sequential(front,se,backbone).to(DEVICE)
    head=RegressionHead(512,meta_dim=3,hetero=True).to(DEVICE)
    opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),
                          lr=3e-4,weight_decay=WD)
    scaler=torch.cuda.amp.GradScaler()
    best={"NRMSE":9e9}

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr = train_epoch(model, head, train_loader, opt, scaler, hetero=True)
        mae, mse, r2, nrmse = evaluate(model, head, val_loader, hetero=True)
        print(f"[Ep{ep:02d}] TrainLoss={tr:.4f} | Val MSE={mse:.4f} MAE={mae:.4f} R2={r2:.3f} time={time.time()-t0:.1f}s")

        # ✅ MSE 기준으로 저장 (NRMSE 제외)
        if mse < best["NRMSE"]:   # key 이름 그대로 두되 값은 MSE
            best = {"NRMSE": float(mse), "epoch": ep}  # float32 → float 변환
            torch.save({
                "front": front.state_dict(),
                "backbone": backbone.state_dict(),
                "head": head.state_dict()
            }, "best_cbramod_raw_finetune.pth")

            # ✅ JSON 직렬화 가능한 타입으로 기록
            with open("best_metrics.json", "w") as f:
                json.dump({
                    "MAE": float(mae),
                    "MSE": float(mse),
                    "R2": float(r2)
                }, f, indent=2)

    print("✅ Best:", best)


if __name__=="__main__":
    main()
