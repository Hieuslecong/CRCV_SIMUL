from pathlib import Path
import argparse,pickle,json,torch,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
from crcv52.geometry import GeoMLP
from crcv52.field_b import CenterlineFieldNetB,infer_candidates_multitarget_topk_b
from crcv52.recovery import source_proposals
from crcv52.dual_recovery import ranked_endpoint_infos
from crcv52.v54_verifier import prepare_context,descriptor,source_key,add_relative_features
from crcv52.gap_metrics import core_gap_hit
A=ROOT/'artifacts';geo=GeoMLP();geo.load_state_dict(torch.load(A/'models/geometry_xy.pt',map_location='cpu'));geo.eval();field=CenterlineFieldNetB(c=16);field.load_state_dict(torch.load(A/'models/centerline_field_v52b.pt',map_location='cpu')['state_dict']);field.eval();epr=pickle.load(open(A/'models/endpoint_ranker_v53.pkl','rb'))

def unpack(r):
 bits=np.frombuffer(r['add_pack'],np.uint8);return np.unpackbits(bits)[:np.prod(r['shape'])].reshape(r['shape']).astype(bool)

def run(split):
 src=A/'cache'/f'v54_{split}_coregap_parts';dst=A/'cache'/f'v54_{split}_coregap_topk_parts';dst.mkdir(exist_ok=True);basefn={'module':'v52_module.pkl','cal':'v52_cal.pkl'}[split];rows={r['name']:r for r in pickle.load(open(A/'cache'/basefn,'rb'))}
 for pp in sorted(src.glob('*.pkl')):
  outp=dst/pp.name
  if outp.exists():continue
  d=pickle.load(open(pp,'rb'));r=rows[d['stat']['image']];ctx=prepare_context(r);base=r['base'].astype(bool);gt=r['gt'].astype(bool);im=r['image'].astype(np.float32);p=r['prob'].astype(np.float32);thr=float(r['threshold'])
  # Trim old relative features; they are recomputed jointly after adding the top-k family.
  C=[];seen=set()
  for z in d['records']:
   zz=dict(z);zz['feature']=np.asarray(zz['feature'][:-8],np.float32);C.append(zz);seen.add((zz['family'],zz['add_pack']))
  infos=ranked_endpoint_infos(r,epr,per_component=2,max_sources=12);top=[z.endpoint_yx for z in infos];qs=source_proposals(geo,base,max_sources=12,horizons=(6,12,18,24),angles=(-8,0,8),corridor_radius=6,endpoint_subset=top);raw=infer_candidates_multitarget_topk_b(field,im,p,base,thr,qs,batch=128,targets_per_band=3,nms_radius=4)
  added=0
  for z in raw:
   a=z['add'].astype(bool);pack=np.packbits(a.reshape(-1)).tobytes();key=('v52b_topk',pack)
   if key in seen:continue
   seen.add(key);tp=int((a&gt).sum());fp=int((a&~gt).sum());pr=tp/(tp+fp+1e-9);hit,core,_=core_gap_hit(a,gt,base,2,1);lab=1 if pr>=.85 and hit>=2 else (0 if pr<=.20 or hit==0 else -1);f=descriptor(r,'v52b',z,ctx);ev=float(z.get('mean_field',0.));C.append({'image':r['name'],'typ':r['typ'],'family':'v52b_topk','source_key':source_key('v52b',z),'feature':f,'label':lab,'meta':{'exact_precision':pr,'tp':tp,'fp':fp,'core_gap_hit':hit,'core_gap_total':int(core.sum()),'clearance':2},'add_pack':pack,'shape':a.shape,'raw_score':float(z.get('score',0.)),'evidence':ev,'length':float(a.sum()),'source_score':0.});added+=1
  C=add_relative_features(C);st={'image':r['name'],'typ':r['typ'],'n':len(C),'pos':sum(x['label']==1 for x in C),'neg':sum(x['label']==0 for x in C),'amb':sum(x['label']<0 for x in C),'topk_added':added};pickle.dump({'records':C,'stat':st},open(outp,'wb'),protocol=pickle.HIGHEST_PROTOCOL);print(split,st,flush=True)
 stats=[];N=P=N0=AMB=0
 for pth in sorted(dst.glob('*.pkl')):
  st=pickle.load(open(pth,'rb'))['stat'];stats.append(st);N+=st['n'];P+=st['pos'];N0+=st['neg'];AMB+=st['amb']
 rep={'split':split,'semantics':'core_gap+v52b_topk','n':N,'pos':P,'neg':N0,'amb':AMB,'stats':stats};(A/'results'/f'v54_{split}_coregap_topk_bank.json').write_text(json.dumps(rep,indent=2));print(json.dumps({k:v for k,v in rep.items() if k!='stats'},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['module','cal'],required=True);args=ap.parse_args();run(args.split)
