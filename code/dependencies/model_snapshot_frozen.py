# -*- coding: utf-8 -*-
"""算法变体筛选实验：V0基线→V6全组合，KIMORE(5x5折)+UI-PRMD(10x10折LOSO)，同协议同预算"""
import json, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

DATA = "/root/msgt/data/RehabPile"; DEV = "cuda"
KIMORE = ["LA","LT","PR","Sq","TR"]; UIPRMD = ["DS","HS","IL","SASLR","SL","SSA","SSE","SSIER","SSS","STS"]
torch.manual_seed(42); np.random.seed(42)

def load_action_meta(act):
    info = json.load(open(f"{DATA}/{act}/info.json"))
    kt = info["kinematic_tree"]; J = info["n_joints"]
    A = np.zeros((J,J)); par = np.zeros(J, dtype=int)
    for chain in kt:
        for a,b in zip(chain[:-1],chain[1:]): A[a,b]=A[b,a]=1; par[b]=a
    return J, A, par

def load_fold(act, fold):
    p = f"{DATA}/{act}/fold{fold}"
    def lx(f):
        a = np.load(f, allow_pickle=True).astype(np.float32); N,D,T = a.shape; J = D//3
        return a.transpose(0,2,1).reshape(N,T,J,3)
    xtr, xte = lx(f"{p}/x_train_fold{fold}.npy"), lx(f"{p}/x_test_fold{fold}.npy")
    ytr = np.load(f"{p}/y_train_fold{fold}.npy", allow_pickle=True).astype(np.float32).ravel()
    yte = np.load(f"{p}/y_test_fold{fold}.npy", allow_pickle=True).astype(np.float32).ravel()
    if ytr.max() > 1.5: ytr, yte = ytr/100., yte/100.
    return xtr, np.clip(ytr,0,1), xte, np.clip(yte,0,1)

def norm_bones(x, par):
    # x:(N,T,J,3) 减关节0+除躯干长；再拼骨骼单位向量 -> (N,T,J,6)
    x = x - x[:,:, :1, :]
    d = np.linalg.norm(x[:,:,3,:], axis=-1).mean(axis=1)[:,None,None,None]
    x = x / np.clip(d, 1e-6, None)
    N,T,J,_ = x.shape
    bone = x - x[:,:,par,:]          # (N,T,J,3)
    nb = np.linalg.norm(bone, axis=-1, keepdims=True)
    bone = bone / np.clip(nb, 1e-6, None); bone[:,:,0]=0
    return np.concatenate([x, bone], axis=-1)

class Net(nn.Module):
    def __init__(self, J, A, cin=3, C=64, multiscale=False, transformer=False, adaptA=False, dist=False):
        super().__init__()
        I = torch.eye(J); Ah = torch.tensor(A,dtype=torch.float32)+I
        Dm = Ah.sum(1,keepdim=True).pow(-0.5)
        self.register_buffer("A", (Dm*Ah*Dm.T))
        self.adaptA = adaptA; self.J = J
        if adaptA:
            self.emb = nn.Parameter(torch.randn(J, 16)*0.1)
            self.adapt_bn = nn.BatchNorm1d(J)
        self.g1 = nn.Conv2d(cin, C, 1)
        self.multiscale = multiscale
        if multiscale:
            self.tc1 = nn.ModuleList([nn.Conv2d(C,C,(k,1),padding=(k//2,0)) for k in (3,5,9)])
            self.tc2 = nn.ModuleList([nn.Conv2d(C,C*2,(k,1),stride=(2,1),padding=(k//2,0)) for k in (3,5,9)])
        else:
            self.tc1 = nn.Conv2d(C,C,(9,1),padding=(4,0))
            self.tc2 = nn.Conv2d(C,C*2,(9,1),stride=(2,1),padding=(4,0))
        self.gc1, self.gc2 = nn.Conv2d(C,C,1), nn.Conv2d(C,C*2,1)
        self.bn1, self.bn2 = nn.BatchNorm2d(C), nn.BatchNorm2d(C*2)
        self.transformer = transformer
        if transformer:
            self.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=C*2, nhead=8, dim_feedforward=C*4, batch_first=True, dropout=0.1), num_layers=2)
        self.dist = dist
        self.head = nn.Linear(C*2, KB if dist else 1)
    def gconv(self, x):
        if not self.adaptA: return x @ self.A
        B = self.emb @ self.emb.T / 16.0
        B = F.relu(B); B = B / (B.sum(1, keepdim=True)+1e-6)
        return x @ (self.A + B)
    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.g1(x)
        x = self.bn1(F.relu((self.tc1(self.gconv(x)) if not self.multiscale else sum(t(self.gconv(x)) for t in self.tc1))))
        x = self.gconv(x)
        x = self.bn2(F.relu((self.tc2(x) if not self.multiscale else sum(t(x) for t in self.tc2))))
        if self.transformer:
            x = x.mean(dim=3).transpose(1,2)   # (B,T,C2)
            x = self.tr(x).mean(dim=1)
        else:
            x = x.mean(dim=(2,3))
        return self.head(x).squeeze(-1)

KB = 10; CENTERS = torch.tensor([(i+0.5)/KB for i in range(KB)])
def run_config(name, cfg, acts, folds, use_bones, dist_loss):
    J, A, par = load_action_meta(acts[0])
    all_mae, all_r, all_rho, all_base = [], [], [], []
    for act in acts:
        info = json.load(open(f"{DATA}/{act}/info.json"))
        J, A, par = load_action_meta(act)
        import glob as _g
        folds = len(_g.glob(f"{DATA}/{act}/fold*"))  # 动态折数
        for fold in range(folds):
            xtr,ytr,xte,yte = load_fold(act, fold)
            base = np.abs(yte - ytr.mean()).mean()
            if use_bones: xtr, xte = norm_bones(xtr,par), norm_bones(xte,par)
            else:
                xtr = (lambda x: (x - x[:,:,:1,:]) / np.clip(np.linalg.norm(x[:,:,3,:],axis=-1).mean(1)[:,None,None,None],1e-6,None))(xtr)
                xte = (lambda x: (x - x[:,:,:1,:]) / np.clip(np.linalg.norm(x[:,:,3,:],axis=-1).mean(1)[:,None,None,None],1e-6,None))(xte)
            cin = 6 if use_bones else 3
            torch.manual_seed(42); model = Net(J, A, cin=cin, dist=dist_loss, **cfg).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            xtrT = torch.tensor(xtr).to(DEV); ytrT = torch.tensor(ytr).to(DEV)
            xteT = torch.tensor(xte).to(DEV); n = len(xtrT)
            ctr = CENTERS.to(DEV)
            for ep in range(60):
                model.train()
                perm = torch.randperm(n, device=DEV)
                for i in range(0, n, 32):
                    idx = perm[i:i+32]
                    out = model(xtrT[idx])
                    if dist_loss:
                        bins = torch.clamp((ytrT[idx]*KB).long(), 0, KB-1)
                        loss = F.cross_entropy(out, bins)
                    else:
                        loss = F.mse_loss(torch.sigmoid(out), ytrT[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                if dist_loss: pr = (torch.softmax(model(xteT), dim=1) @ ctr.unsqueeze(1)).squeeze(-1).cpu().numpy()
                else: pr = torch.sigmoid(model(xteT)).cpu().numpy()
            all_mae.append(np.abs(pr-yte).mean()); all_base.append(base)
            try: all_r.append(pearsonr(pr,yte)[0])
            except Exception: all_r.append(np.nan)
        print(f"  [{name}|{act}] MAE={np.mean(all_mae[-folds:]):.4f} r={np.nanmean(all_r[-folds:]):.3f}", flush=True)
    return dict(name=name, mae=float(np.mean(all_mae)), mae_std=float(np.std(all_mae)),
                pearson=float(np.nanmean(all_r)), trivial=float(np.mean(all_base)),
                n_train_runs=len(all_mae))

CONFIGS = [
    ("V0_基线STGCN",      dict(multiscale=False,transformer=False,adaptA=False), False, False),
    ("V1_多尺度",          dict(multiscale=True, transformer=False,adaptA=False), False, False),
    ("V2_时序Transformer", dict(multiscale=False,transformer=True, adaptA=False), False, False),
    ("V3_骨骼特征",        dict(multiscale=False,transformer=False,adaptA=False), True,  False),
    ("V4_结构组合",        dict(multiscale=True, transformer=True, adaptA=False), True,  False),
    ("V5_分布回归",        dict(multiscale=True, transformer=True, adaptA=False), True,  True),
    ("V6_自适应图",        dict(multiscale=True, transformer=True, adaptA=True),  True,  True),
]
if __name__ == "__main__":
    out = {}
    for name, cfg, bones, distl in CONFIGS:
        t0 = time.time()
        print(f"===== {name} @KIMORE =====", flush=True)
        out[f"{name}_KIMORE"] = run_config(name, cfg, KIMORE, 5, bones, distl)
        print(f"===== {name} @UIPRMD =====", flush=True)
        out[f"{name}_UIPRMD"] = run_config(name, cfg, UIPRMD, 10, bones, distl)
        out[f"{name}_KIMORE"]["time_s"] = round(time.time()-t0, 1)
        json.dump(out, open("/root/msgt/variants_results.json","w"), indent=2, ensure_ascii=False)
    print("ALL-DONE", flush=True)
