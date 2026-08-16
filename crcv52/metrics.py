from __future__ import annotations
import cv2,numpy as np
from skimage.morphology import skeletonize
def binary(mask,gt):
 a=mask.astype(bool);g=gt.astype(bool);tp=int((a&g).sum());fp=int((a&~g).sum());fn=int((~a&g).sum());return {'dice':2*tp/(2*tp+fp+fn+1e-9),'iou':tp/(tp+fp+fn+1e-9),'precision':tp/(tp+fp+1e-9),'recall':tp/(tp+fn+1e-9)}
def cldice(mask,gt):
 a=mask.astype(bool);g=gt.astype(bool);sa=skeletonize(a);sg=skeletonize(g);tprec=(sa&g).sum()/(sa.sum()+1e-9);tsens=(sg&a).sum()/(sg.sum()+1e-9);return 2*tprec*tsens/(tprec+tsens+1e-9)
def ncomp(m):return max(cv2.connectedComponents(m.astype(np.uint8),8)[0]-1,0)
