##########################################################################
# EEG Foundation Challenge 2025 – Unified Submission (CBraMod)
# Compatible with Codabench scoring interface
# ---------------------------------------------------------------
# Challenge 1 : CBraMod → CCD RT regression
# Challenge 2 : CBraMod → p-factor regression
##########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# ============================================================
# CBraMod components
# ============================================================
class SincConv1d(nn.Module):
    def __init__(self, out_channels=64, kernel_size=129, sample_rate=100,
                 min_hz=0.3, max_hz=45.0):
        super().__init__()
        self.out_channels, self.kernel_size = out_channels, kernel_size
        self.sample_rate, self.min_hz, self.max_hz = sample_rate, min_hz, max_hz
        low = torch.linspace(min_hz, max_hz - 5.0, out_channels)
        band = torch.ones(out_channels) * 5.0
        self.low_hz_ = nn.Parameter(low)
        self.band_hz_ = nn.Parameter(band)
        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        self.register_buffer("n", n)

    def forward(self, x):
        B, C, T = x.shape
        device, dtype = x.device, x.dtype
        low = torch.clamp(torch.abs(self.low_hz_), min=self.min_hz,
                          max=self.max_hz - 1)
        high = torch.clamp(low + torch.abs(self.band_hz_),
                           min=low + 1.0,
                           max=torch.full_like(low, self.max_hz))
        n = self.n.to(device=device, dtype=dtype)
        window = torch.hamming_window(self.kernel_size, periodic=False,
                                      dtype=dtype, device=device)
        nyq = self.sample_rate / 2.0
        filters = []
        for i in range(self.out_channels):
            f1, f2 = low[i]/nyq, high[i]/nyq
            h1 = 2*f2*torch.sinc(2*f2*n)
            h2 = 2*f1*torch.sinc(2*f1*n)
            filters.append((h1-h2)*window)
        filt = torch.stack(filters, dim=0).unsqueeze(1)
        x_dw = x.view(B*C,1,T)
        y = F.conv1d(x_dw,filt,stride=1,padding=self.kernel_size//2)
        y = y.view(B,C,self.out_channels,y.shape[-1]).sum(dim=1)
        return y


class SEBlock(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(c, c//r), nn.Linear(c//r, c)
    def forward(self, x):
        s = x.mean(-1)
        e = torch.sigmoid(self.fc2(F.relu(self.fc1(s)))).unsqueeze(-1)
        return x * e


class CBraModBackbone(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(64,128,7,padding=3), nn.ReLU(),
            nn.Conv1d(128,256,5,padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256,out_dim)
    def forward(self,x):
        return self.fc(self.conv(x).squeeze(-1))


class RtHead(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim,max(64,feat_dim//2)),
            nn.ReLU(),
            nn.Linear(max(64,feat_dim//2),1)
        )
    def forward(self,z):
        if z.ndim>2: z=torch.flatten(z,1)
        return self.mlp(z)


class RegressionHead(nn.Module):
    def __init__(self,in_dim,meta_dim=3):
        super().__init__()
        self.meta_fc = nn.Sequential(nn.Linear(meta_dim,32),nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(in_dim+32,128),
                                nn.ReLU(),nn.Linear(128,1))
    def forward(self,f,meta):
        m = self.meta_fc(meta)
        return self.fc(torch.cat([f,m],1)).squeeze(-1)

# ============================================================
# Submission class (SERVER INTERFACE)
# ============================================================
class Submission:
    """
    EEG Challenge 2025 Submission (Codabench compatible)
    - get_model_challenge_1(): CBraMod→CCD RT regression
    - get_model_challenge_2(): CBraMod→p-factor regression
    """
    def __init__(self, sfreq=100, device=None):
        if isinstance(sfreq, torch.device):   # 잘못된 순서로 들어온 경우 자동 교정
            sfreq, device = 100, sfreq
        self.sfreq = sfreq
        self.device = device or torch.device("cpu")

    # ---------------- Challenge 1 ----------------
    def get_model_challenge_1(self):
        weight_path = os.path.join(os.path.dirname(__file__), "weights_ch1.pth")

        front = SincConv1d(64,129,self.sfreq)
        se = SEBlock(64)
        backbone = CBraModBackbone(512)
        head = RtHead(512)

        try:
            ckpt = torch.load(weight_path, map_location=self.device)

            enc_state = ckpt["encoder"]
            front.load_state_dict(
                {k:v for k,v in enc_state.items()
                 if "low_hz" in k or "band_hz" in k}, strict=False)
            backbone.load_state_dict(enc_state, strict=False)
            head.load_state_dict(ckpt["rt_head"], strict=False)
            print("[INFO] Loaded Challenge 1 weights (weights_ch1.pth)")
        except Exception as e:
            print(f"[WARN] Could not load Challenge 1 weights: {e}")

        model = nn.Sequential(front, se, backbone, head).to(self.device)
        model.eval()
        return model

    # ---------------- Challenge 2 ----------------
    def get_model_challenge_2(self):
        weight_path = os.path.join(os.path.dirname(__file__), "weights_ch2.pth")

        front = SincConv1d(64,129,self.sfreq)
        se = SEBlock(64)
        backbone = CBraModBackbone(512)
        head = RegressionHead(512, meta_dim=3)

        try:
            ckpt = torch.load(weight_path, map_location=self.device)
            
            front.load_state_dict(ckpt["front"], strict=False)
            backbone.load_state_dict(ckpt["backbone"], strict=False)
            head.load_state_dict(ckpt["head"], strict=False)
            print("[INFO] Loaded Challenge 2 weights (weights_ch2.pth)")
        except Exception as e:
            print(f"[WARN] Could not load Challenge 2 weights: {e}")

        class CBraModFull(nn.Module):
            def __init__(self, front, se, backbone, head):
                super().__init__()
                self.front, self.se, self.backbone, self.head = \
                    front, se, backbone, head
            def forward(self, x, meta=None):
                if meta is None:
                    meta = torch.zeros(x.size(0),3,device=x.device)
                z = self.backbone(self.se(self.front(x)))
                return self.head(z, meta)

        model = CBraModFull(front, se, backbone, head).to(self.device)
        model.eval()
        return model
