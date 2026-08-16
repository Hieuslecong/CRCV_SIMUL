from __future__ import annotations
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class GeoMLP(nn.Module):
    def __init__(self,hist=6,future=6):super().__init__();self.hist=hist;self.future=future;self.net=nn.Sequential(nn.Linear((hist-1)*2,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,future*2))
    def forward(self,x):return self.net(x).reshape(-1,self.future,2)
def parse_xy(path):
    seq=[];all=[]
    for raw in open(path):
        s=raw.strip()
        if not s:continue
        try:a,b=map(float,s.split(',')[:2])
        except:continue
        if abs(a)<1e-12 and abs(b)<1e-12:
            if len(seq)>=8:all.append(np.asarray(seq,np.float32))
            seq=[]
        else:seq.append((a,b))
    if len(seq)>=8:all.append(np.asarray(seq,np.float32))
    return all
def _norm_hist_future(seq,t,h=6,f=6):
    H=seq[t-h+1:t+1].copy();Fut=seq[t+1:t+1+f].copy();o=H[-1].copy();H-=o;Fut-=o;v=H[-1]-H[-2];ang=np.arctan2(v[1],v[0]);ca,sa=np.cos(-ang),np.sin(-ang);R=np.array([[ca,-sa],[sa,ca]],np.float32);H=H@R.T;Fut=Fut@R.T;scale=max(np.mean(np.linalg.norm(np.diff(H,axis=0),axis=1)),1e-5);return (np.diff(H,axis=0)/scale).reshape(-1),Fut/scale,scale,ang,o
def train_geometry(path,epochs=40,seed=5151,h=6,f=6,max_samples=12000):
    rng=np.random.default_rng(seed);seqs=parse_xy(path);X=[];Y=[]
    for s in seqs:
        for t in range(h-1,len(s)-f):
            x,y,*_=_norm_hist_future(s,t,h,f);X.append(x);Y.append(y)
    if len(X)>max_samples:
        q=rng.choice(len(X),max_samples,replace=False);X=np.asarray(X)[q];Y=np.asarray(Y)[q]
    else:X=np.asarray(X);Y=np.asarray(Y)
    torch.manual_seed(seed);m=GeoMLP(h,f);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);xt=torch.tensor(X);yt=torch.tensor(Y)
    for ep in range(epochs):
        ids=rng.permutation(len(X));m.train()
        for st in range(0,len(ids),256):
            q=ids[st:st+256];z=m(xt[q]);loss=F.smooth_l1_loss(z,yt[q]);opt.zero_grad();loss.backward();opt.step()
    m.eval();return m,{'trajectories':len(seqs),'samples':len(X)}
def predict_future(m,hist,total=24):
    hist=np.asarray(hist,np.float32);out=[];cur=hist.copy()
    while len(out)<total:
        H=cur[-m.hist:];o=H[-1].copy();v=H[-1]-H[-2];ang=np.arctan2(v[1],v[0]);ca,sa=np.cos(-ang),np.sin(-ang);R=np.array([[ca,-sa],[sa,ca]],np.float32);Hr=(H-o)@R.T;scale=max(np.mean(np.linalg.norm(np.diff(Hr,axis=0),axis=1)),1e-5);x=(np.diff(Hr,axis=0)/scale).reshape(1,-1)
        with torch.no_grad():pr=m(torch.tensor(x,dtype=torch.float32))[0].numpy()*scale
        ca,sa=np.cos(ang),np.sin(ang);Ri=np.array([[ca,-sa],[sa,ca]],np.float32);pr=pr@Ri.T+o;need=min(len(pr),total-len(out));chunk=pr[:need];out.extend(chunk.tolist());cur=np.vstack([cur,chunk])[-m.hist:]
    return np.asarray(out,np.float32)
