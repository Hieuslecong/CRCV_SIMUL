import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from crcv_q1.protocol import Q1Protocol,hash_protocol
from crcv_q1.stats import bootstrap_ci,paired_permutation_p,holm_bonferroni
from crcv_q1.gates import assess

def good_payload():
 return {
  'datasets':['A','B','C'],'backbones':['a','b','c','d','e'],'resolutions':[128,256],
  'full_training_seeds':[1337,2027,31415], 'cross_dataset_routes':['A->B','B->C'],
  'comparators':['base','morphology','crcv_v5181'], 'lobo':{'completed':True}, 'official_backbone':True,
  'latency':{'cpu':True,'edge':True}, 'final_test_state':'SEALED','external':{'datasets_completed':1},
  'aggregate':{'mean_delta_f1':.015,'mean_delta_miou':.007,'positive_pair_rate':.9,'worst_delta_f1':-.001},
  'statistics':{'f1':{'ci_low':.003,'holm_p':.01},'miou':{'ci_low':.001,'holm_p':.02}}
 }

def test_protocol_hash_stable():
 p=Q1Protocol().validate(); assert hash_protocol(p)==hash_protocol(p)
def test_stats_positive():
 b=[.2,.3,.4,.5];r=[.22,.32,.43,.52];assert bootstrap_ci(b,r,n_boot=1000)['ci_low']>0;assert paired_permutation_p(b,r,n_perm=2000)['p']<.2
def test_holm_monotone_bounds():
 a=holm_bonferroni([.01,.02,.2]); assert all(0<=x<=1 for x in a)
def test_good_payload_passes(): assert assess(good_payload())['status']=='Q1_READY'
def test_missing_external_blocks():
 x=good_payload();x['external']['datasets_completed']=0;assert assess(x)['status']=='BLOCKED'
def test_unsealed_test_blocks():
 x=good_payload();x['final_test_state']='READ_DURING_DEV';assert assess(x)['status']=='BLOCKED'
