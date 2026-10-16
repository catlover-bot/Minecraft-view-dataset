import math
import numpy as np
import pytest
from offline_evaluation.analysis.goal010v2 import independent as x


def test_boundary_and_exact_distance():
    g={(a,b,c):'stone' for a in range(3) for b in range(3) for c in range(3)}
    assert len(x.boundary(g))==26
    p={(a+1,b,c):'stone' for a,b,c in g}
    result=x.inspect(g,p)
    assert result['surface_prediction_hits']==[26,26]
    assert result['surface_gt_hits']==[26,26]
    assert result['C']==18


def test_dense_and_offset_methods_agree(monkeypatch):
    a={(-3,0,1),(0,0,0),(1,1,1),(12,0,-3)}; b={(0,0,0),(-1,0,1),(15,0,-3)}
    dense=x.surface_counts(a,b)[0]
    monkeypatch.setattr(x,'MAX_EDT_CELLS',0)
    assert x.surface_counts(a,b)[0]==dense
    assert dense==[1,3]


def test_empty_negative_floor_and_column_union():
    g={(-1,0,-1):'glass',(-1,2,-1):'glass'}
    result=x.inspect(g,{})
    assert result['fp_union']==1 and result['column_absolute_error_sum']==2
    assert result['pooled']['2_0']['iou']==0
    assert x._pooled({(-1,0,0)},{(0,0,0)},2,0)['intersection']==0
    assert x._pooled({(-1,0,0)},{(0,0,0)},2,1)['intersection']==1


def test_random_surface_matches_bruteforce():
    rng=np.random.default_rng(20261017)
    for _ in range(10):
        a={tuple(map(int,c)) for c in rng.integers(-4,5,size=(12,3))}
        b={tuple(map(int,c)) for c in rng.integers(-4,5,size=(9,3))}
        counts,_=x.surface_counts(a,b)
        assert counts==[sum(min(sum((c[i]-d[i])**2 for i in range(3)) for d in b)<=t*t for c in a) for t in (1,2)]
