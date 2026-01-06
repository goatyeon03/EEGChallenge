# ============================================================
# EEG Foundation Challenge 2025 - Challenge 1
# CBraMod pretrained encoder → SSL (non-CCD raw EEG) → CCD RT regression
# ============================================================

import os, random, numpy as np, pandas as pd, warnings, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob
import torch.nn.functional as F
from collections import defaultdict
import mne
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BIDS_ROOT         = "/data5/open_data/HBN/EEG_BIDS/"
PREPROCESSED_ROOT = "/data5/open_data/HBN/Preprocessed_EEG/0922try_bySubject/"
CBRAMOD_WEIGHTS   = "/home/RA/EEG_Challenge/Challenge2/CBraMod/pretrained_weights/pretrained_weights.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SFREQ = 100
N_CHANS = 19
SAMPLE_RATIO = 0.2
WIN_S = 2.0
BATCH_SSL, BATCH_CCD = 16, 8
EPOCHS_SSL, EPOCHS_CCD = 5, 5
LR_SSL, LR_CCD = 1e-4, 3e-4
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ============================================================
# EEG LOADER (원본 .set 사용)
# ============================================================
def read_raw_eeg(eeg_path):
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
    raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False)
    raw.resample(SFREQ)
    X = raw.get_data(picks="eeg").astype(np.float32)
    X = np.nan_to_num((X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6))
    return X

def make_window(x, center_s, win_s=WIN_S, sfreq=SFREQ):
    t1 = int(center_s * sfreq)
    Tw = int(win_s * sfreq)
    t0 = max(0, t1 - Tw)
    seg = x[:, t0:t1]
    need = Tw - seg.shape[1]
    if need > 0:
        seg = np.pad(seg, ((0,0),(need,0)), mode="constant")
    return seg.astype(np.float32)

# ============================================================
# SAMPLER
# ============================================================
def sample_balanced_files(file_label_pairs, ratio=0.2):
    """task별 균등 샘플링"""
    task_to_files = defaultdict(list)
    for f, t in file_label_pairs:
        task_to_files[t].append(f)
    sampled=[]
    for t, files in task_to_files.items():
        k=max(1,int(len(files)*ratio))
        sampled+=random.sample(files,k)
        print(f"[INFO] {t}: {k}/{len(files)} ({100*k/len(files):.1f}%)")
    print(f"[INFO] Total sampled: {len(sampled)} files")
    return [(f,t) for f,t in file_label_pairs if f in sampled]

# ============================================================
# DATASETS
# ============================================================
class SSL_Dataset(Dataset):
    """non-CCD EEG self-supervised"""
    def __init__(self, file_label_pairs):
        self.items=[]
        for f,_ in file_label_pairs:
            X=read_raw_eeg(f)
            T=X.shape[1]
            for t1 in range(200, T, int(SFREQ)):
                self.items.append((f, t1/SFREQ))
        random.shuffle(self.items)
    def _aug(self,x):
        x=x+0.01*np.random.randn(*x.shape).astype(np.float32)
        if np.random.rand()<0.5:
            L=max(1,int(x.shape[1]*0.1)); s=np.random.randint(0,x.shape[1]-L+1)
            x[:,s:s+L]=0
        return x
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        f,c=self.items[idx]
        X=read_raw_eeg(f)[:N_CHANS]
        seg=make_window(X,c)
        v1=self._aug(seg.copy()); v2=self._aug(seg.copy())
        return torch.from_numpy(v1), torch.from_numpy(v2)

class CCD_RTDataset(Dataset):
    """CCD RT regression (원본 EEG)"""
    def __init__(self, eeg_event_pairs):
        self.samples=[]
        for eeg_path, ev_path in eeg_event_pairs:
            if not os.path.exists(ev_path): continue
            df=pd.read_csv(ev_path,sep="\t")
            for o,rt in self.extract_ccd_trials(df):
                self.samples.append((eeg_path,o,rt))
        rts=np.array([s[2] for s in self.samples],dtype=np.float32)
        self.rt_mean,self.rt_std=rts.mean(),rts.std()+1e-6
        print(f"✅ CCD Dataset: {len(self.samples)} trials (RT mean={self.rt_mean:.1f})")
    @staticmethod
    def extract_ccd_trials(df):
        if df.empty or "onset" not in df.columns: return []
        on,val=df["onset"].astype(float).values,df["value"].astype(str).values
        fb=df["feedback"].astype(str).values if "feedback" in df.columns else ["n/a"]*len(df)
        starts=[i for i,v in enumerate(val) if "contrastTrial_start" in v]
        presses=[i for i,v in enumerate(val) if "buttonPress" in v]
        trials=[]
        for ti in starts:
            t0=on[ti]; later=[pi for pi in presses if on[pi]>t0]
            if not later: continue
            pi=later[0]; rt=(on[pi]-t0)*1000
            if 100<=rt<=3000 and "smiley" in fb[pi].lower():
                trials.append((t0,rt))
        return trials
    def __len__(self): return len(self.samples)
    def __getitem__(self,idx):
        p,o,rt=self.samples[idx]
        X=read_raw_eeg(p)[:N_CHANS]
        seg=make_window(X,o)
        rt_norm=(rt-self.rt_mean)/self.rt_std
        return torch.from_numpy(seg), torch.tensor([rt_norm],dtype=torch.float32)

# ============================================================
# MODEL
# ============================================================
class CBraModEncoder(nn.Module):
    def __init__(self,embed_dim=256,n_layers=4):
        super().__init__()
        self.frontend=nn.Sequential(
            nn.Conv1d(N_CHANS,64,3,padding=1),nn.BatchNorm1d(64),nn.GELU(),
            nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.GELU(),
            nn.Conv1d(128,embed_dim,3,padding=1))
        layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=8,
                                         dim_feedforward=512,dropout=0.1,
                                         activation="gelu",batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,n_layers)
        self.pool=nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        z=self.frontend(x)
        z=z.transpose(1,2)
        z=self.transformer(z)
        z=z.transpose(1,2)
        return self.pool(z).squeeze(-1)

class RtHead(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(in_dim,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x): return self.fc(x)

# ============================================================
# TRAIN
# ============================================================
def contrastive_loss(z1,z2,temp=0.5):
    z1=F.normalize(z1,dim=1); z2=F.normalize(z2,dim=1)
    sim=torch.mm(z1,z2.T)/temp
    labels=torch.arange(z1.size(0),device=z1.device)
    loss=(F.cross_entropy(sim,labels)+F.cross_entropy(sim.T,labels))/2
    return loss

def train_ssl(dl,encoder,epochs=EPOCHS_SSL,lr=LR_SSL):
    feat_dim=encoder(torch.randn(1,N_CHANS,200).to(DEVICE)).shape[1]
    proj=nn.Sequential(nn.Linear(feat_dim,feat_dim//2),nn.ReLU(),nn.Linear(feat_dim//2,128)).to(DEVICE)
    opt=torch.optim.AdamW(list(encoder.parameters())+list(proj.parameters()),lr=lr)
    for ep in range(epochs):
        encoder.train(); proj.train(); losses=[]
        for x1,x2 in tqdm(dl,desc=f"[SSL] Epoch {ep+1}/{epochs}"):
            x1,x2=x1.to(DEVICE),x2.to(DEVICE)
            z1,z2=encoder(x1),encoder(x2)
            p1,p2=proj(z1),proj(z2)
            loss=contrastive_loss(p1,p2)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {ep+1}: Contrastive Loss={np.mean(losses):.4f}")
    torch.save(encoder.state_dict(),"encoder_ssl_cbramod_raw.pth")
    print("✅ Saved encoder_ssl_cbramod_raw.pth")
    return encoder

def train_ccd_rt(dl,encoder,rt_head,epochs=EPOCHS_CCD,lr=LR_CCD):
    opt=torch.optim.AdamW(rt_head.parameters(),lr=lr)
    loss_fn=nn.L1Loss(); encoder.eval()
    for ep in range(epochs):
        rt_head.train(); losses=[]
        for x,y in tqdm(dl,desc=f"[CCD] Epoch {ep+1}/{epochs}"):
            x,y=x.to(DEVICE),y.to(DEVICE)
            with torch.no_grad(): feat=encoder(x)
            pred=rt_head(feat).squeeze(); loss=loss_fn(pred,y.squeeze())
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"Epoch {ep+1}: MAE={np.mean(losses):.4f}")
    torch.save({"encoder":encoder.state_dict(),"rt_head":rt_head.state_dict()},
               "weights_ch1_cbramod_raw.pth")
    print("✅ Saved weights_ch1_cbramod_raw.pth")

# ============================================================
# MAIN
# ============================================================
def main():
    # ---- non-CCD (SSL) ----
    preproc_files=[]
    for dirpath,_,files in os.walk(PREPROCESSED_ROOT):
        for fn in files:
            if not fn.endswith("_eeg_pp.set"): continue
            f=os.path.join(dirpath,fn)
            if "contrastChangeDetection" in f.lower(): continue
            task="unknown"
            for t in ["RestingState", "DespicableMe", "FunWithFractals",
                "ThePresent",
                "DiaryOfAWimpyKid",
                "contrastChangeDetection",
                "surroundSupp",
                "seqLearning6target",
                "seqLearning8target",
                "symbolSearch",]:
                if t.lower() in f.lower(): task=t
            preproc_files.append((f,task))
    sampled=sample_balanced_files(preproc_files,SAMPLE_RATIO)
    ds_ssl=SSL_Dataset(sampled)
    dl_ssl=DataLoader(ds_ssl,batch_size=BATCH_SSL,shuffle=True,num_workers=4,pin_memory=True)
    # pretrained CBraMod encoder
    encoder=CBraModEncoder().to(DEVICE)
    state=torch.load(CBRAMOD_WEIGHTS,map_location=DEVICE)
    if "encoder" in state: state=state["encoder"]
    encoder.load_state_dict(state,strict=False)
    print("✅ Loaded pretrained CBraMod encoder.")
    # SSL stage
    encoder=train_ssl(dl_ssl,encoder,epochs=EPOCHS_SSL,lr=LR_SSL)

    # ---- CCD (RT regression) ----
    ev_files=glob(os.path.join(BIDS_ROOT,"ds*/sub-*","eeg","sub-*_task-contrastChangeDetection*_events.tsv"))
    ccd_files=[]
    for dirpath,_,files in os.walk(PREPROCESSED_ROOT):
        for fn in files:
            if "contrastChangeDetection" in fn and fn.endswith("_eeg_pp.set"):
                f=os.path.join(dirpath,fn)
                subj=os.path.basename(f).split("_")[0]
                match=[e for e in ev_files if subj in e]
                if match: ccd_files.append((f,match[0]))
    print(f"🔗 Matched {len(ccd_files)} CCD EEG↔event files.")
    ds_ccd=CCD_RTDataset(ccd_files)
    dl_ccd=DataLoader(ds_ccd,batch_size=BATCH_CCD,shuffle=True,num_workers=4,pin_memory=True)
    with torch.no_grad():
        feat_dim=encoder(torch.randn(1,N_CHANS,200).to(DEVICE)).shape[1]
    rt_head=RtHead(feat_dim).to(DEVICE)
    train_ccd_rt(dl_ccd,encoder,rt_head,EPOCHS_CCD,LR_CCD)

if __name__=="__main__":
    main()
