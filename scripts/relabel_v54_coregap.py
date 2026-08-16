from pathlib import Path
import argparse,pickle,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
from crcv52.gap_metrics import core_gap_hit
A=ROOT/'artifacts'

def unpack(r):
 bits=np.frombuffer(r['add_pack'],dtype=np.uint8);return np.unpackbits(bits)[:np.prod(r['shape'])].reshape(r['shape']).astype(bool)

def run(split):
 src=A/'cache'/('v54_module_bank_compact' if split=='module' else 'v54_cal_bank_parts');dst=A/'cache'/f'v54_{split}_coregap_parts';dst.mkdir(exist_ok=True)
 basefn={'module':'v52_module.pkl','cal':'v52_cal.pkl'}[split];base={r['name']:r for r in pickle.load(open(A/'cache'/basefn,'rb'))};stats=[];N=P=N0=AMB=0
 for p in sorted(src.glob('*.pkl')):
  d=pickle.load(open(p,'rb'));out=[]
  for r in d['records']:
   rr=dict(r);br=base[r['image']];a=unpack(r);gt=br['gt'].astype(bool);b=br['base'].astype(bool);tp=int((a&gt).sum());fp=int((a&~gt).sum());pr=tp/(tp+fp+1e-9);hit,core,_=core_gap_hit(a,gt,b,clearance=2,tolerance=1)
   lab=1 if pr>=.85 and hit>=2 else (0 if pr<=.20 or hit==0 else -1);rr['label']=lab;rr['meta']=dict(rr['meta']);rr['meta'].update({'core_gap_hit':hit,'core_gap_total':int(core.sum()),'exact_precision':pr,'clearance':2});out.append(rr)
  st={'image':d['stat']['image'],'typ':d['stat']['typ'],'n':len(out),'pos':sum(x['label']==1 for x in out),'neg':sum(x['label']==0 for x in out),'amb':sum(x['label']<0 for x in out)};pickle.dump({'records':out,'stat':st},open(dst/p.name,'wb'),protocol=pickle.HIGHEST_PROTOCOL);stats.append(st);N+=st['n'];P+=st['pos'];N0+=st['neg'];AMB+=st['amb'];print(split,st,flush=True)
 rep={'split':split,'semantics':'core_gap_clearance_2px','n':N,'pos':P,'neg':N0,'amb':AMB,'stats':stats};(A/'results'/f'v54_{split}_coregap_bank.json').write_text(json.dumps(rep,indent=2));print(json.dumps({k:v for k,v in rep.items() if k!='stats'},indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['module','cal'],required=True);args=ap.parse_args();run(args.split)
