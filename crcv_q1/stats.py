from __future__ import annotations
import numpy as np

PRIMARY=("precision","recall","f1","miou")

def paired_delta(base, refined):
    b=np.asarray(base,dtype=float); r=np.asarray(refined,dtype=float)
    if b.shape != r.shape or b.ndim != 1 or len(b)<2:
        raise ValueError("base/refined must be same-length 1-D arrays with n>=2")
    if not (np.isfinite(b).all() and np.isfinite(r).all()):
        raise ValueError("metrics must be finite")
    return r-b

def bootstrap_ci(base, refined, seed=1337, n_boot=10000, alpha=0.05):
    d=paired_delta(base,refined); rng=np.random.default_rng(seed); n=len(d)
    idx=rng.integers(0,n,size=(int(n_boot),n))
    means=d[idx].mean(axis=1)
    lo,hi=np.quantile(means,[alpha/2,1-alpha/2])
    return {"mean":float(d.mean()),"ci_low":float(lo),"ci_high":float(hi),"n":int(n)}

def paired_permutation_p(base, refined, seed=1337, n_perm=20000, alternative="greater"):
    d=paired_delta(base,refined); obs=float(d.mean()); rng=np.random.default_rng(seed)
    signs=rng.choice(np.array([-1.,1.]),size=(int(n_perm),len(d)))
    null=(signs*d).mean(axis=1)
    if alternative=="greater": p=(np.count_nonzero(null>=obs)+1)/(len(null)+1)
    elif alternative=="two-sided": p=(np.count_nonzero(np.abs(null)>=abs(obs))+1)/(len(null)+1)
    else: raise ValueError("alternative must be greater or two-sided")
    return {"mean_delta":obs,"p":float(p),"n_perm":int(n_perm),"alternative":alternative}

def holm_bonferroni(p_values):
    p=np.asarray(p_values,float)
    if p.ndim!=1 or np.any((p<0)|(p>1)): raise ValueError("invalid p-values")
    order=np.argsort(p); m=len(p); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=min(1.0,(m-rank)*p[idx]); running=max(running,val); adj[idx]=running
    return adj.tolist()
