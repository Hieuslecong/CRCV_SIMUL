from __future__ import annotations
import numpy as np

def annotate_consensus(row):
    groups={}
    for z in row['candidates']:
        key=tuple(z.get('source_endpoint_yx',z.get('source_yx')))
        groups.setdefault(key,[]).append(z)
    for key,gs in groups.items():
        if not gs:continue
        shape=gs[0]['add'].shape;support=np.zeros(shape,np.float32)
        for z in gs:support+=z['add'].astype(np.float32)
        denom=max(len(gs),1)
        # group-relative scalar distributions for core beam signals
        core=['score','min_identity','stop_prob','mean_field','min_field','length']
        vals={k:np.asarray([float(z[k]) for z in gs],np.float32) for k in core}
        med={k:float(np.median(v)) for k,v in vals.items()};sd={k:float(np.std(v)+1e-6) for k,v in vals.items()}
        for z in gs:
            m=z['add'].astype(bool);sv=support[m]/denom if m.any() else np.asarray([0.],np.float32)
            z['consensus_mean']=float(sv.mean());z['consensus_min']=float(sv.min());z['source_candidate_count']=len(gs)
            for k in core:z['rel_'+k]=(float(z[k])-med[k])/sd[k]
    return row
