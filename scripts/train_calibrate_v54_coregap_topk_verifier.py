from pathlib import Path
import pickle,json,numpy as np,cv2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from skimage.morphology import skeletonize
from crcv52.gap_metrics import core_missing_skeleton,core_gap_hit
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts'

def load_parts(split,compact_module=True):
 d=A/'cache'/f'v54_{split}_coregap_topk_parts';rows=[]
 for p in sorted(d.glob('*.pkl')):rows+=pickle.load(open(p,'rb'))['records']
 return rows

def unpack(r):
 bits=np.frombuffer(r['add_pack'],dtype=np.uint8);a=np.unpackbits(bits)[:np.prod(r['shape'])].reshape(r['shape']).astype(bool);return a

M=load_parts('module');C=load_parts('cal',False)
# Normalize source key with image scope to avoid groups crossing images.
for r in M:r['gkey']=(r['image'],r['source_key'])
for r in C:r['gkey']=(r['image'],r['source_key'])
valid=[r for r in M if r['label']>=0];dim=len(valid[0]['feature']);groups={}
for r in valid:groups.setdefault(r['gkey'],[]).append(r)
# Hard individual bank: all positives + strongest same-source negatives + strongest normal negatives.
hard=[];pairs=[];rng=np.random.default_rng(5401)
for g,rs in groups.items():
 pos=[r for r in rs if r['label']==1];neg=[r for r in rs if r['label']==0]
 hard+=pos
 if pos:
  # same-source hard negatives by candidate evidence/raw score
  negs=sorted(neg,key=lambda r:(r['raw_score']+.7*r['evidence']+.25*r['source_score']),reverse=True)[:max(12,3*len(pos))]
  hard+=negs
  # Pair each positive against up to eight hard negatives. Add both orientations.
  for p in pos:
   for n in negs[:8]:
    d=p['feature']-n['feature'];pairs.append((d,1));pairs.append((-d,0))
 else:
  # Source with no positives contributes only a few most deceptive negatives.
  hard+=sorted(neg,key=lambda r:(r['raw_score']+.7*r['evidence']+.25*r['source_score']),reverse=True)[:4]
# Deduplicate individual records by object identity is not stable across lists; feature/label duplicates are harmless but cap.
X=np.stack([r['feature'] for r in hard]).astype(np.float32);y=np.array([r['label'] for r in hard],np.int64)
sc=StandardScaler().fit(X);Xs=sc.transform(X);absclf=LogisticRegression(C=.5,class_weight='balanced',max_iter=2000,random_state=5402).fit(Xs,y)
if pairs:
 PX=np.stack([z[0] for z in pairs]).astype(np.float32);Py=np.array([z[1] for z in pairs],np.int64);psc=StandardScaler().fit(PX);pairclf=LogisticRegression(C=.5,class_weight='balanced',max_iter=2000,random_state=5403).fit(psc.transform(PX),Py)
else:raise RuntimeError('no positive same-source pairs')

def raw_score(r):
 f=r['feature'][None].astype(np.float32);abslog=float(absclf.decision_function(sc.transform(f))[0]);# ranking utility uses pair scaler + learned linear direction
 z=psc.transform(f);util=float((z@pairclf.coef_[0]).item());return .5*abslog+.5*util
# candidate-level CAL diagnostic only; image-level selection below is decisive.
Cv=[r for r in C if r['label']>=0];cy=np.array([r['label'] for r in Cv]);cs=np.array([raw_score(r) for r in Cv]);diag={'auc':float(roc_auc_score(cy,cs)) if len(set(cy))>1 else None,'ap':float(average_precision_score(cy,cs)) if len(set(cy))>1 else None,'n':len(cy),'pos':int(cy.sum())}
calbase={r['name']:r for r in pickle.load(open(A/'cache/v52_cal.pkl','rb'))}
# score and retain top candidate per same-source group before global acceptance.
byimg={}
for r in C:
 r['_score']=raw_score(r);byimg.setdefault(r['image'],[]).append(r)

def eval_thr(th,max_accept=2):
 TP=FP=MSK=SH=normal=accepted=0;details=[]
 for name,rs in byimg.items():
  rr=calbase[name];base=rr['base'].astype(bool);gt=rr['gt'].astype(bool);typ=rr['typ'];best={}
  for r in rs:
   if r['_score']<th:continue
   g=r['source_key']
   if g not in best or r['_score']>best[g]['_score']:best[g]=r
  cand=sorted(best.values(),key=lambda r:r['_score'],reverse=True);out=base.copy();add=np.zeros_like(base,bool);acc=[]
  for r in cand:
   a=unpack(r)&~out
   if int(a.sum())<2 or np.any(a&add):continue
   out[a]=1;add|=a;acc.append(r)
   if len(acc)>=max_accept:break
  if typ=='normal':normal+=int(add.sum());details.append({'image':name,'normal_added':int(add.sum()),'accepted':len(acc)});continue
  tp=int((add&gt).sum());fp=int((add&~gt).sum());ms=core_missing_skeleton(gt,base,clearance=2);tol=cv2.dilate(add.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool);sh=int((tol&ms).sum());TP+=tp;FP+=fp;MSK+=int(ms.sum());SH+=sh;accepted+=len(acc);details.append({'image':name,'tp':tp,'fp':fp,'skeleton_hit':sh,'accepted':len(acc),'families':[r['family'] for r in acc]})
 return {'threshold':float(th),'added_precision':TP/(TP+FP+1e-9),'missing_skeleton_recovery':SH/(MSK+1e-9),'tp':TP,'fp':FP,'normal_added':normal,'accepted':accepted,'details':details}
# threshold grid comes from CAL scores only, no model/hyperparameter changes.
vals=np.unique(np.quantile(cs,np.linspace(.70,.999,180)))
res=[eval_thr(float(t),2) for t in vals];feas=[m for m in res if m['added_precision']>=.85 and m['normal_added']==0 and m['accepted']>0]
if feas:best=max(feas,key=lambda m:(m['missing_skeleton_recovery'],m['added_precision'],-m['threshold']));qualified=True
else:best=max(res,key=lambda m:(m['added_precision'],m['missing_skeleton_recovery'],-m['fp']));qualified=False
ck={'schema':'crcv-v5.4-coregap-topk-pairwise-verifier-1','scaler':sc,'absclf':absclf,'pair_scaler':psc,'pairclf':pairclf,'threshold':best['threshold'],'max_accept':2,'qualified_on_calibration':qualified,'feature_dim':dim}
pickle.dump(ck,open(A/'models/verifier_v54_coregap_topk_pairwise.pkl','wb'))
rep={'schema':ck['schema'],'module_hard_n':len(hard),'module_hard_pos':int(y.sum()),'pair_samples':len(pairs),'cal_candidate_diag':diag,'calibration':best,'qualified':qualified};(A/'results/verifier_v54_coregap_topk_calibration.json').write_text(json.dumps(rep,indent=2));print(json.dumps({**{k:v for k,v in rep.items() if k!='calibration'},'calibration':{k:v for k,v in best.items() if k!='details'}},indent=2))
