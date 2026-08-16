from pathlib import Path
import argparse,pickle,json,torch,cv2,numpy as np
from skimage.morphology import skeletonize
ROOT=Path(__file__).resolve().parents[1]
from crcv52.geometry import GeoMLP
from crcv52.recovery import source_proposals
from crcv52.field_b import CenterlineFieldNetB,infer_candidates_multitarget_b
from crcv52.dual_recovery import ranked_endpoint_infos,generate_connect_family,ridge_continue_candidates
A=ROOT/'artifacts';PART=A/'cache/v54_oracle_parts';PART.mkdir(exist_ok=True)
geo=GeoMLP();geo.load_state_dict(torch.load(A/'models/geometry_xy.pt',map_location='cpu'));geo.eval();field=CenterlineFieldNetB(c=16);field.load_state_dict(torch.load(A/'models/centerline_field_v52b.pt',map_location='cpu')['state_dict']);field.eval();epr=pickle.load(open(A/'models/endpoint_ranker_v53.pkl','rb'));rows=pickle.load(open(A/'cache/v52_real_val.pkl','rb'))

def lab(r,add):
 gt=r['gt'].astype(bool);base=r['base'].astype(bool);a=add.astype(bool);tp=int((a&gt).sum());fp=int((a&~gt).sum());prec=tp/(tp+fp+1e-9);ms=skeletonize(gt)&~base;tol=cv2.dilate(a.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool);hit=tol&ms;return prec,tp,fp,int(hit.sum()),hit

def compress(zs,n=96):
 pool=[]
 for key in ['score','mean_ridge','min_ridge','length']:
  pool += sorted(zs,key=lambda z:float(z.get(key,0.)),reverse=True)[:n//4]
 seen=set();out=[]
 for z in pool:
  k=np.packbits(z['add']).tobytes()
  if k in seen:continue
  seen.add(k);out.append(z)
 return out[:n]

def process(i):
 r=rows[i];im=r['image'].astype(np.float32);p=r['prob'].astype(np.float32);b=r['base'].astype(bool);thr=float(r['threshold']);infos=ranked_endpoint_infos(r,epr,per_component=2,max_sources=12);top=[z.endpoint_yx for z in infos];pool=[]
 qs=source_proposals(geo,b,max_sources=12,horizons=(6,12,18,24),angles=(-8,0,8),corridor_radius=6,endpoint_subset=top)
 for z in infer_candidates_multitarget_b(field,im,p,b,thr,qs,batch=128):pool.append(('v52b',z['add'],z))
 part=A/'cache/iterative_parts_val'/f'{i:03d}.pkl'
 if part.exists():
  rr=pickle.load(open(part,'rb'))
  for z in rr['candidates']:pool.append(('iterative',z['add'],z))
 for info in infos:
  zs=compress(ridge_continue_candidates(r,info,geo,beam_width=12,branch_k=4,max_steps=72),96)
  for z in zs:pool.append(('ridge_continue',z['add'],z))
 conn,_,pairs=generate_connect_family(r,epr,field,geo,max_sources=12,per_component=2,max_pairs=36)
 for z in conn:pool.append(('bidir_connect',z['add'],z))
 P=[]
 for fam,a,z in pool:
  prec,tp,fp,sh,hit=lab(r,a)
  if prec>=.85 and sh>=2:P.append({'family':fam,'add':a.astype(bool),'precision':prec,'tp':tp,'fp':fp,'sh':sh,'hit':hit})
 selected=[];ids=set();covered=np.zeros_like(b,bool);added=np.zeros_like(b,bool)
 for _ in range(5):
  cand=[]
  for q in P:
   if id(q) in ids:continue
   marginal=int((q['hit']&~covered).sum())
   if marginal<=0:continue
   aa=q['add']&~added;mtp=int((aa&r['gt'].astype(bool)).sum());mfp=int((aa&~r['gt'].astype(bool)).sum());mp=mtp/(mtp+mfp+1e-9)
   if aa.any() and mp<.75:continue
   cand.append((marginal,q['precision'],mtp-mfp,q))
  if not cand:break
  q=max(cand,key=lambda x:(x[0],x[1],x[2]))[-1];selected.append(q);ids.add(id(q));covered|=q['hit'];added|=q['add']
 prec,tp,fp,sh,hit=lab(r,added);ms=skeletonize(r['gt'].astype(bool))&~b
 out={'index':i,'image':r['name'],'pool':len(pool),'pairs':len(pairs),'positive_pool':len(P),'selected_families':[q['family'] for q in selected],'tp':tp,'fp':fp,'precision':prec,'skeleton_hit':sh,'missing_skeleton':int(ms.sum()),'per_family_positive':{f:sum(q['family']==f for q in P) for f in ['v52b','iterative','ridge_continue','bidir_connect']}}
 (PART/f'{i:03d}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=10);args=ap.parse_args()
 for i in range(args.start,min(args.end,10)):
  p=PART/f'{i:03d}.json'
  if p.exists():print('skip',i);continue
  process(i)
 # aggregate available parts
 arr=[]
 for i in range(10):
  p=PART/f'{i:03d}.json'
  if p.exists():arr.append(json.loads(p.read_text()))
 if len(arr)==10:
  TP=sum(x['tp'] for x in arr);FP=sum(x['fp'] for x in arr);SH=sum(x['skeleton_hit'] for x in arr);MS=sum(x['missing_skeleton'] for x in arr);rep={'version':'crcv-v5.4-dual-mode-hybrid-oracle-2','precision':TP/(TP+FP+1e-9),'missing_skeleton_recovery':SH/(MS+1e-9),'positive_images':sum(x['positive_pool']>0 for x in arr),'coverage':sum(x['positive_pool']>0 for x in arr)/10,'tp':TP,'fp':FP,'skeleton_hit':SH,'missing_skeleton':MS,'details':arr};(A/'results/hybrid_proposal_oracle_v54.json').write_text(json.dumps(rep,indent=2));print('FINAL',json.dumps({k:v for k,v in rep.items() if k!='details'},indent=2))
