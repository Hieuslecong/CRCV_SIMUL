from pathlib import Path
import pickle,json,cv2,numpy as np,torch
from skimage.morphology import skeletonize
ROOT=Path(__file__).resolve().parents[1]
from crcv52.geometry import GeoMLP
from crcv52.field_b import CenterlineFieldNetB
from crcv52.dual_recovery import generate_connect_family
from crcv52.verifier import label_candidate
A=ROOT/'artifacts'
geo=GeoMLP();geo.load_state_dict(torch.load(A/'models/geometry_xy.pt',map_location='cpu'));geo.eval()
field=CenterlineFieldNetB(c=16);field.load_state_dict(torch.load(A/'models/centerline_field_v52b.pt',map_location='cpu')['state_dict']);field.eval()
epr=pickle.load(open(A/'models/endpoint_ranker_v53.pkl','rb'))

def eval_split(rows):
 TP=FP=SH=MSK=0;posimg=0;details=[]
 for r in rows:
  if r['typ']!='crack':continue
  cands,infos,pairs=generate_connect_family(r,epr,field,geo,max_sources=16,per_component=2,max_pairs=48)
  P=[]
  for z in cands:
   zz={'add':z['add'],'path':z['path'],'candidate':{'source_yx':z['source_yx'],'history':np.asarray(z.get('history',[[z['source_yx'][1]-1,z['source_yx'][0]],[z['source_yx'][1],z['source_yx'][0]]]),np.float32),'corridor':z['corridor'],'horizon':24,'angle':0},'score':z['pair_score'],'mean_field':z['mean_pair_field'],'min_field':z['min_pair_field'],'target_score':z['mean_pair_field'],'field_margin':0.}
   # exact label logic locally to avoid fake history dependency
   gt=r['gt'].astype(bool);base=r['base'].astype(bool);add=z['add'].astype(bool);tp=int((add&gt).sum());fp=int((add&~gt).sum());prec=tp/(tp+fp+1e-9);ms=skeletonize(gt)&~base;tol=cv2.dilate(add.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool);sh=int((tol&ms).sum())
   if prec>=.85 and sh>=2:P.append(({'precision':prec,'tp':tp,'fp':fp,'skeleton_hit':sh},z))
  posimg+=int(bool(P));P=sorted(P,key=lambda x:(x[0]['skeleton_hit'],x[0]['precision'],x[0]['tp']),reverse=True);add=np.zeros_like(r['base'],bool);used=[]
  for m,z in P:
   aa=z['add']&~add
   if aa.sum()<2:continue
   add|=z['add'];used.append({'variant':z['variant'],'pair_distance':z['pair_distance'],'facing_a':z['facing_a'],'facing_b':z['facing_b'],'source':z['source_endpoint_yx'],'dest':z['destination_endpoint_yx']})
   if len(used)>=3:break
  gt=r['gt'].astype(bool);base=r['base'].astype(bool);ms=skeletonize(gt)&~base;tol=cv2.dilate(add.astype(np.uint8),np.ones((3,3),np.uint8),1).astype(bool);tp=int((add&gt).sum());fp=int((add&~gt).sum());sh=int((tol&ms).sum());TP+=tp;FP+=fp;SH+=sh;MSK+=int(ms.sum())
  details.append({'image':r['name'],'endpoint_infos':len(infos),'pairs':len(pairs),'candidates':len(cands),'positive_candidates':len(P),'tp':tp,'fp':fp,'skeleton_hit':sh,'missing_skeleton':int(ms.sum()),'used':used})
  print(r['name'],'infos',len(infos),'pairs',len(pairs),'cand',len(cands),'pos',len(P),'hit',sh,'tp/fp',tp,fp,flush=True)
 return {'precision':TP/(TP+FP+1e-9),'missing_skeleton_recovery':SH/(MSK+1e-9),'positive_images':posimg,'crack_images':sum(r['typ']=='crack' for r in rows),'tp':TP,'fp':FP,'details':details}

if __name__=='__main__':
 rows=pickle.load(open(A/'cache/v52_real_val.pkl','rb'))
 rep=eval_split(rows)
 (A/'results/dual_connect_oracle_v54.json').write_text(json.dumps(rep,indent=2))
 print(json.dumps({k:v for k,v in rep.items() if k!='details'},indent=2))
