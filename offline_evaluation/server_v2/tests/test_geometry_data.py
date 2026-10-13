import json
import numpy as np
import pytest

from offline_evaluation.server_v2.core import ValidationError
from offline_evaluation.server_v2.geometry_data import load_geometry, geometry_metrics, storage_truncation_evidence


def saved(tmp_path, labels):
    a=np.asarray(labels,dtype='<U32').reshape(1,1,len(labels))
    np.save(tmp_path/'voxels.npy',a)
    (tmp_path/'bbox.json').write_text(json.dumps(dict(xmin=-2,xmax=-2,ymin=7,ymax=7,zmin=3,zmax=2+len(labels))))
    return tmp_path/'voxels.npy',tmp_path/'bbox.json'


def test_ambiguous_material_keeps_occupancy_without_fake_material_grid(tmp_path):
    read=load_geometry(*saved(tmp_path,['air','stone','quartz_(or_white_block)']))
    assert read.geometry_valid and not read.material_valid and read.material_grid is None
    assert read.coordinates=={(-2,7,4),(-2,7,5)}
    assert read.invalid_labels[0]['material']=='UNKNOWN'
    metrics=geometry_metrics(read.coordinates,read.coordinates)
    assert metrics['occupancy_iou']==1 and metrics['dimension_mae']==0
    assert metrics['material_aware_iou'] is metrics['total_normalized_repair_cost'] is None


@pytest.mark.parametrize('label',['dark_wood_or_nether_brick','stonebrick_mossy_or_cobble_mix'])
def test_identifier_syntax_does_not_certify_a_descriptive_material_alternative(tmp_path,label):
    read=load_geometry(*saved(tmp_path,[label]))
    assert read.geometry_valid and not read.material_valid and read.material_grid is None
    assert read.invalid_labels[0]['classification']=='MULTIPLE_OR_UNCERTAIN_MATERIAL'
    assert read.coordinates=={(-2,7,3)}


@pytest.mark.parametrize('label',['stone|air','air?','','stone_(or_air)','stone_or_air'])
def test_unknown_air_presence_prevents_complete_geometry_scoring(tmp_path,label):
    read=load_geometry(*saved(tmp_path,[label]))
    assert not read.geometry_valid and not read.material_valid
    assert not read.coordinates


def test_mixed_uncertain_presence_keeps_only_definite_cells(tmp_path):
    read=load_geometry(*saved(tmp_path,['stone_or_air','stairs','stone']))
    assert not read.geometry_valid and read.coordinates=={(-2,7,4),(-2,7,5)}


def test_exact_air_variants_and_world_axis_mapping(tmp_path):
    read=load_geometry(*saved(tmp_path,['minecraft:air','cave_air','void_air','minecraft:stone']))
    assert read.geometry_valid and read.material_valid
    assert read.material_grid.voxels=={(-2,7,6):'stone'}


def test_truncated_label_preserved_and_source_not_changed(tmp_path):
    paths=saved(tmp_path,['sandstone_(or_light_yellow_block)'])
    before=paths[0].read_bytes()
    read=load_geometry(*paths)
    assert read.invalid_labels[0]['stored_label']=='sandstone_(or_light_yellow_block'
    assert paths[0].read_bytes()==before and read.material_grid is None
    assert storage_truncation_evidence(read,{'operations':[{'block':'sandstone_(or_light_yellow_block)'}]})[0]['capacity']==32


def test_syntactically_valid_truncated_material_is_still_unknown(tmp_path):
    full='a'*33
    read=load_geometry(*saved(tmp_path,[full]))
    assert read.material_valid
    evidence=storage_truncation_evidence(read,{'operations':[{'block':full}]})
    assert evidence and not read.material_valid and read.material_grid is None


def test_air_alternative_hidden_beyond_storage_capacity_is_geometry_unknown(tmp_path):
    full='a'*32+'_or_air'
    read=load_geometry(*saved(tmp_path,[full,'stone']))
    evidence=storage_truncation_evidence(read,{'operations':[{'block':full}]})
    assert evidence and not read.geometry_valid
    assert read.coordinates=={(-2,7,4)}


def test_asymmetric_yxz_storage_maps_to_negative_world_origin(tmp_path):
    a=np.full((2,3,4),'air',dtype='<U32')
    a[1,2,3]='quartz_(or_white_block)'
    np.save(tmp_path/'voxels.npy',a)
    (tmp_path/'bbox.json').write_text(json.dumps(dict(xmin=-12,xmax=-10,ymin=8,ymax=9,zmin=-9,zmax=-6)))
    read=load_geometry(tmp_path/'voxels.npy',tmp_path/'bbox.json')
    assert read.geometry_valid and not read.material_valid
    assert read.coordinates=={(-10,9,-6)}


def test_object_pickle_remains_forbidden(tmp_path):
    paths=saved(tmp_path,['stone'])
    np.save(paths[0],np.asarray([[['stone']]],dtype=object))
    with pytest.raises(ValidationError,match='pickle'):load_geometry(*paths)
