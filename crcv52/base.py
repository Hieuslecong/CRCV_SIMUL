from __future__ import annotations
import random, math
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from .data import load_item,Item

class DSConv(nn.Module):
    def __init__(self,ci,co,s=1):
        super().__init__();self.dw=nn.Conv2d(ci,ci,3,s,1,groups=ci,bias=False);self.pw=nn.Conv2d(ci,co,1,bias=False);self.gn=nn.GroupNorm(min(8,co),co)
    def forward(self,x):return F.silu(self.gn(self.pw(self.dw(x))))
class Block(nn.Module):
    def __init__(self,ci,co):super().__init__();self.a=DSConv(ci,co);self.b=DSConv(co,co)
    def forward(self,x):return self.b(self.a(x))
class TinyUNet(nn.Module):
    def __init__(self,c=16):
        super().__init__();self.e1=Block(3,c);self.e2=Block(c,c*2);self.e3=Block(c*2,c*4);self.b=Block(c*4,c*6);self.d3=Block(c*6+c*4,c*4);self.d2=Block(c*4+c*2,c*2);self.d1=Block(c*2+c,c);self.o=nn.Conv2d(c,1,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2));z=self.b(F.max_pool2d(c,2));z=F.interpolate(z,c.shape[-2:],mode='bilinear',align_corners=False);z=self.d3(torch.cat([z,c],1));z=F.interpolate(z,b.shape[-2:],mode='bilinear',align_corners=False);z=self.d2(torch.cat([z,b],1));z=F.interpolate(z,a.shape[-2:],mode='bilinear',align_corners=False);z=self.d1(torch.cat([z,a],1));return self.o(z)[:,0]
class SegDS(Dataset):
    def __init__(self,items,size=128,augment=True):self.items=items;self.size=size;self.aug=augment
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        im,m=load_item(self.items[i],self.size)
        if self.aug:
            if random.random()<.5:im=im[:,::-1].copy();m=m[:,::-1].copy()
            if random.random()<.5:im=im[::-1].copy();m=m[::-1].copy()
            if random.random()<.3:
                g=random.uniform(.85,1.15);im=np.clip(im**g,0,1)
        return torch.tensor(im.transpose(2,0,1),dtype=torch.float32),torch.tensor(m,dtype=torch.float32)
def loss_fn(logit,y):
    pos=y.sum();neg=y.numel()-pos;pw=torch.clamp(neg/(pos+1),1,18);b=F.binary_cross_entropy_with_logits(logit,y,pos_weight=pw);p=torch.sigmoid(logit);dice=1-(2*(p*y).sum((1,2))+1)/((p+y).sum((1,2))+1);return b+.8*dice.mean()
def train_model(model,items,epochs=5,seed=1,size=128,lr=1e-3,batch=12):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(min(8,torch.get_num_threads()));ds=SegDS(items,size,True);dl=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=0);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=2e-4)
    for ep in range(epochs):
        model.train();tot=0
        for x,y in dl:
            z=model(x);loss=loss_fn(z,y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();tot+=float(loss)*len(x)
    model.eval();return model
@torch.no_grad()
def predict(model,image,out_size=256,model_size=128):
    im=cv2.resize(image,(model_size,model_size),interpolation=cv2.INTER_AREA) if image.shape[:2]!=(model_size,model_size) else image
    x=torch.tensor(im.transpose(2,0,1)[None],dtype=torch.float32);p=torch.sigmoid(model(x))[0].numpy();return cv2.resize(p,(out_size,out_size),interpolation=cv2.INTER_LINEAR).astype(np.float32)
def metrics(mask,gt):
    a=mask.astype(bool);g=gt.astype(bool);tp=(a&g).sum();fp=(a&~g).sum();fn=(~a&g).sum();d=2*tp/(2*tp+fp+fn+1e-9);iou=tp/(tp+fp+fn+1e-9);pr=tp/(tp+fp+1e-9);re=tp/(tp+fn+1e-9);return d,iou,pr,re
def select_threshold(model,items,grid=None):
    if grid is None:grid=np.linspace(.15,.85,29)
    cache=[]
    for it in items:
        im,m=load_item(it,256);cache.append((predict(model,im),m))
    rows=[]
    for t in grid:
        ds=[metrics(p>=t,m)[0] for p,m in cache];rows.append((float(np.mean(ds)),float(t)))
    return max(rows)[1],max(rows)[0]
