from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv, hashlib
import cv2, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
DBG = DATA/'debug/dataset'
REAL = DATA/'real/real_debug_data'

@dataclass(frozen=True)
class Item:
    name: str
    image_path: Path
    mask_path: Path|None
    split: str
    source: str
    typ: str
    lineage: str


def _norm01_rgb(path: Path, size: int|None=None):
    im=cv2.cvtColor(cv2.imread(str(path),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    if im is None: raise FileNotFoundError(path)
    if size is not None and im.shape[:2]!=(size,size): im=cv2.resize(im,(size,size),interpolation=cv2.INTER_AREA)
    return im.astype(np.float32)/255.

def _mask(path: Path|None, size: int|None=None, shape=None):
    if path is None:
        if shape is None: raise ValueError('shape required for normal mask')
        return np.zeros(shape,np.uint8)
    m=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(path)
    if size is not None and m.shape!=(size,size): m=cv2.resize(m,(size,size),interpolation=cv2.INTER_NEAREST)
    return (m>127).astype(np.uint8)

def load_item(it: Item, size: int|None=None):
    im=_norm01_rgb(it.image_path,size)
    m=_mask(it.mask_path,size,im.shape[:2])
    return im,m

def debug_items(split='train', require_native_match=True):
    out=[]
    for name in [x.strip() for x in open(DBG/f'{split}.txt') if x.strip()]:
        ip=DBG/'Images'/name; mp=DBG/'Labels'/name
        if not ip.exists() or not mp.exists(): continue
        if require_native_match and Image.open(ip).size!=Image.open(mp).size: continue
        stem=Path(name).stem
        if stem.startswith('CRACK500_'): lin=stem
        elif stem.startswith('DeepCrack_'): lin=stem.rsplit('-',1)[0]
        elif stem.startswith('Khanh11k_'): lin='_'.join(stem.split('_')[:4])
        elif stem.startswith('Masonry_'): lin='_'.join(stem.split('_')[:3])
        else: lin=stem
        out.append(Item(name,ip,mp,f'debug_{split}','debug','crack',lin))
    return out

def real_items(split='train'):
    rows=list(csv.DictReader(open(REAL/'manifest.csv')));out=[]
    for r in rows:
        if r['split']!=split: continue
        if r['image_path']:
            n=r['image_name']; out.append(Item(n,REAL/r['image_path'],REAL/r['mask_path'],f'real_{split}','real','crack',Path(n).stem.replace('training_','').replace('validation_','')))
        elif r['normal_path']:
            n=r['normal_name']; out.append(Item(n,REAL/r['normal_path'],None,f'real_{split}','real','normal',Path(n).stem))
    return out

def group_folds(items:list[Item], k=3, salt='v52-oof'):
    folds=[set() for _ in range(k)]
    for typ in sorted(set(x.typ for x in items)):
        gs=sorted(set(x.lineage for x in items if x.typ==typ),key=lambda g:hashlib.sha1((salt+g).encode()).hexdigest())
        for j,g in enumerate(gs): folds[j%k].add(g)
    return folds

def audit_summary():
    from PIL import Image
    rep={}
    for sp in ['train','val']:
        names=[x.strip() for x in open(DBG/f'{sp}.txt') if x.strip()];bad=[]
        for n in names:
            ip=DBG/'Images'/n;mp=DBG/'Labels'/n
            if not ip.exists() or not mp.exists(): bad.append((n,'missing'))
            elif Image.open(ip).size!=Image.open(mp).size: bad.append((n,Image.open(ip).size,Image.open(mp).size))
        rep[f'debug_{sp}']={'listed':len(names),'usable_native_match':len(names)-len(bad),'excluded_registration_size_mismatch':len(bad)}
    for sp in ['train','val']:
        its=real_items(sp);bad=[]
        for it in its:
            if it.mask_path is not None and Image.open(it.image_path).size!=Image.open(it.mask_path).size:bad.append(it.name)
        rep[f'real_{sp}']={'items':len(its),'crack':sum(x.typ=='crack' for x in its),'normal':sum(x.typ=='normal' for x in its),'registration_mismatch':len(bad)}
    return rep
