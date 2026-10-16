"""Independent integer/set and dense EDT counterchecks, no alignment optimizer."""
from collections import Counter
from itertools import groupby
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure

OFFSETS6 = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
MAX_EDT_CELLS = 8_000_000


def boundary(points):
    # This count-neighbor construction is separate from the metric engine.
    counts = Counter()
    for x,y,z in points:
        for dx,dy,dz in OFFSETS6:
            counts[x+dx,y+dy,z+dz] += 1
    return {p for p in points if counts[p] != 6}


def surface_counts(a, b):
    """Return exact within1/2 matches, independent of KD nearest-neighbor query."""
    if not a or not b:
        return [0,0], 'EMPTY'
    aa=np.asarray(sorted(a),dtype=np.int64); bb=np.asarray(sorted(b),dtype=np.int64)
    lo=np.minimum(aa.min(0),bb.min(0)); hi=np.maximum(aa.max(0),bb.max(0))
    shape=hi-lo+1
    if math.prod(map(int,shape)) <= MAX_EDT_CELLS:
        mask=np.ones(tuple(shape),dtype=bool)
        coords=bb-lo; mask[tuple(coords.T)]=False
        distances=distance_transform_edt(mask)
        observed=distances[tuple((aa-lo).T)]
        return [int(np.count_nonzero(observed<=k)) for k in (1,2)], 'INDEPENDENT_DENSE_EDT'
    counts=[]
    for threshold in (1,2):
        offsets=[(x,y,z) for x in range(-threshold,threshold+1)
            for y in range(-threshold,threshold+1) for z in range(-threshold,threshold+1)
            if x*x+y*y+z*z<=threshold*threshold]
        counts.append(sum(any((x+dx,y+dy,z+dz) in b for dx,dy,dz in offsets) for x,y,z in a))
    return counts, 'INDEPENDENT_FINITE_INTEGER_OFFSETS'


def inspect(g, p):
    gs,ps=set(g),set(p); common=gs&ps
    fp_g={(x,z) for x,y,z in gs}; fp_p={(x,z) for x,y,z in ps}
    cg=Counter((x,z) for x,y,z in gs); cp=Counter((x,z) for x,y,z in ps)
    columns=fp_g|fp_p
    bg,bp=boundary(gs),boundary(ps)
    pc,pm=surface_counts(bp,bg); rc,rm=surface_counts(bg,bp)
    def tops(points):
        ordered=sorted((x,z,y) for x,y,z in points)
        return {column:max(t[2] for t in group) for column,group in groupby(ordered,key=lambda t:t[:2])}
    gt_tops=tops(gs); pred_tops=tops(ps)
    material_counts=Counter(g.values())
    material_correct=Counter(g[c] for c in common if g[c]==p[c])
    return {'g':len(g),'p':len(p),'C':len(common),'M':sum(g[c]==p[c] for c in common),
        'fp_intersection':len(fp_g&fp_p),'fp_union':len(columns),
        'column_absolute_error_sum':sum(abs(cg[c]-cp[c]) for c in columns),
        'gt_columns':len(fp_g),'pred_columns':len(fp_p),
        'common_top_y_absolute_error_sum':sum(abs(gt_tops[c]-pred_tops[c]) for c in fp_g&fp_p),
        'macro_material_recall':sum(material_correct[m]/n for m,n in material_counts.items())/len(material_counts) if material_counts else None,
        'components_g':components(gs),'components_p':components(ps),
        'surface_g':len(bg),'surface_p':len(bp),'surface_prediction_hits':pc,'surface_gt_hits':rc,
        'surface_countercheck_methods':[pm,rm],
        'pooled':{f'{k}_{anchor}': _pooled(gs,ps,k,anchor) for k in (1,2,4,8)
            for anchor in sorted({0,k//2})}}


def components(points):
    if not points: return {'count':0,'largest_fraction':None}
    array=np.asarray(sorted(points),dtype=np.int64); lo=array.min(0); shape=array.max(0)-lo+1
    if math.prod(map(int,shape))<=MAX_EDT_CELLS:
        mask=np.zeros(tuple(shape),dtype=bool); mask[tuple((array-lo).T)]=True
        labels,count=label(mask,structure=generate_binary_structure(3,1))
        sizes=np.bincount(labels[mask]); largest=int(sizes[1:].max())
        return {'count':int(count),'largest_fraction':largest/len(points)}
    # Union-find on positive-axis edges, independently of engine flood fill.
    parents={p:p for p in points}; sizes={p:1 for p in points}
    def find(c):
        while parents[c]!=c:
            parents[c]=parents[parents[c]]; c=parents[c]
        return c
    for p in points:
        for axis in range(3):
            q=tuple(v+(i==axis) for i,v in enumerate(p))
            if q not in points: continue
            a,b=find(p),find(q)
            if a!=b:
                if sizes[a]<sizes[b]: a,b=b,a
                parents[b]=a; sizes[a]+=sizes[b]
    roots={find(p) for p in points}
    return {'count':len(roots),'largest_fraction':max(sizes[r] for r in roots)/len(points)}


def _pooled(g,p,k,anchor):
    a={tuple(math.floor((v-anchor)/k) for v in c) for c in g}
    b={tuple(math.floor((v-anchor)/k) for v in c) for c in p}
    return {'intersection':len(a&b),'union':len(a|b),'iou':len(a&b)/len(a|b) if a|b else 1.0}
