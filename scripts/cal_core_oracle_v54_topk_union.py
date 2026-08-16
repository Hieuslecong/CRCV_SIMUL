from pathlib import Path
import pickle,torch,cv2,numpy as np,json
ROOT=Path(__file__).resolve().parents[1]
from crcv52.geometry import GeoMLP
from crcv52.field_b import CenterlineFieldNetB,infer_candidates_multitarget_topk_b
from crcv52.recovery import source_proposals
from crcv52.dual_recovery import ranked_endpoint_infos
from crcv52.gap_metrics import core_missing_skeleton
A=ROOT/'artifacts';rows=pickle.load(open(A/'cache/v52_cal.pkl','rb'));epr=pickle.load(open(A/'models/endpoint_ranker_v53.pkl','rb'));geo=GeoMLP();geo.load_state_dict(torch.load(A/'models/geometry_xy.pt',map_location='cpu'));geo.eval();field=CenterlineFieldNetB(c=16);field.load_state_dict(torch.load(A/'models/centerline_field_v52b.pt',map_location='cpu')['state_dict']);field.eval();partdir=A/'cache/v54_cal_coregap_parts'
TP=FP=SH=MS=0;posimg=0;details=[]
for i,r in enumerate(rows[:6]):
 b=r['base'].astype(bool);gt=r['gt'].astype(bool);core=core_missing_skeleton(gt,b,2);P=[]
 # existing V5.4 compact candidates
 d=pickle.load(open(partdir/f'{i:03d}.pkl','rb'))
 for z in d['records']:
  if z['label']!=1:continue
  bits=np.frombuffer(z['add_pack'],np.uint8);a=np.unpackbits(bits)[:np.prod(z['shape'])].reshape(z['shape']).astype(bool);tol=cv2.dilate(a.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool);P.append({'fam':z['family'],'a':a,'hit':tol&core,'pr':z['meta']['exact_precision']})
 # new V5.4 top-k target family
 infos=ranked_endpoint_infos(r,epr,per_component=2,max_sources=12);top=[z.endpoint_yx for z in infos];qs=source_proposals(geo,b,max_sources=12,horizons=(6,12,18,24),angles=(-8,0,8),corridor_radius=6,endpoint_subset=top);C=infer_candidates_multitarget_topk_b(field,r['image'].astype(np.float32),r['prob'].astype(np.float32),b,float(r['threshold']),qs,batch=128,targets_per_band=3,nms_radius=4)
 for z in C:
  a=z['add'];tp=int((a&gt).sum());fp=int((a&~gt).sum());pr=tp/(tp+fp+1e-9);tol=cv2.dilate(a.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool);hit=tol&core;sh=int(hit.sum())
  if pr>=.85 and sh>=2:P.append({'fam':'v52b_topk','a':a,'hit':hit,'pr':pr})
 posimg+=int(bool(P));cover=np.zeros_like(b,bool);add=np.zeros_like(b,bool);sel=[];ids=set()
 for _ in range(5):
  Q=[]
  for q in P:
   if id(q) in ids:continue
   m=int((q['hit']&~cover).sum())
   if m<=0:continue
   aa=q['a']&~add;mtp=int((aa&gt).sum());mfp=int((aa&~gt).sum());mp=mtp/(mtp+mfp+1e-9)
   if aa.any() and mp<.75:continue
   Q.append((m,q['pr'],mtp-mfp,q))
  if not Q:break
  q=max(Q,key=lambda x:(x[0],x[1],x[2]))[-1];ids.add(id(q));cover|=q['hit'];add|=q['a'];sel.append(q['fam'])
 tol=cv2.dilate(add.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool);tp=int((add&gt).sum());fp=int((add&~gt).sum());sh=int((tol&core).sum());TP+=tp;FP+=fp;SH+=sh;MS+=int(core.sum());details.append({'image':r['name'],'core_missing':int(core.sum()),'hit':sh,'tp':tp,'fp':fp,'families':sel});print(details[-1])
rep={'semantics':'core_gap_clearance_2px','precision':TP/(TP+FP+1e-9),'core_gap_recovery':SH/(MS+1e-9),'positive_images':posimg,'crack_images':6,'tp':TP,'fp':FP,'core_hit':SH,'core_missing':MS,'details':details};(A/'results/v54_cal_core_oracle_topk_union.json').write_text(json.dumps(rep,indent=2));print(json.dumps({k:v for k,v in rep.items() if k!='details'},indent=2))
