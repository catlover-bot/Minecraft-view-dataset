from offline_evaluation.executor import Block, ExecutionStatus, VoxelGrid, execute


def test_set_and_duplicate_last_write_wins():
    r = execute([{"op":"set","x":1,"y":2,"z":3,"block":"stone"}, {"op":"set","x":1,"y":2,"z":3,"block":"brick"}])
    assert r.status == ExecutionStatus.SUCCESS
    assert r.grid.voxels[(1,2,3)].material == "brick"


def test_fill_is_inclusive_reversed_and_negative():
    r = execute([{"op":"fill","x1":1,"y1":0,"z1":0,"x2":-1,"y2":0,"z2":-1,"block":"stone"}])
    assert len(r.grid.voxels) == 6
    assert (-1,0,-1) in r.grid.voxels and (1,0,0) in r.grid.voxels


def test_carve_and_air_remove():
    r = execute([{"op":"fill","x1":0,"y1":0,"z1":0,"x2":2,"y2":0,"z2":0,"block":"stone"},
                 {"op":"carve","x1":1,"y1":0,"z1":0,"x2":1,"y2":0,"z2":0},
                 {"op":"set","x":2,"y":0,"z":0,"block":"air"}])
    assert set(r.grid.voxels) == {(0,0,0)}


def test_overlapping_commands():
    r = execute([{"op":"fill","x1":0,"y1":0,"z1":0,"x2":1,"y2":1,"z2":1,"block":"stone"},
                 {"op":"fill","x1":1,"y1":1,"z1":1,"x2":2,"y2":2,"z2":2,"block":"glass"}])
    assert len(r.grid.voxels) == 15 and r.grid.voxels[(1,1,1)].material == "glass"


def test_material_normalization():
    r = execute([{"op":"set","x":0,"y":0,"z":0,"block":"minecraft:stone_bricks"},
                 {"op":"set","x":1,"y":0,"z":0,"block":"oak_planks"}])
    assert [r.grid.voxels[(x,0,0)].material for x in range(2)] == ["stonebrick","wood"]


def test_malformed_and_unsupported_are_partial_not_success():
    r = execute([{"op":"set","x":0,"y":0,"z":0,"block":"stone"}, {"op":"teleport","x":1}, {"op":"set","x":"oops","y":0,"z":0,"block":"stone"}])
    assert r.status == ExecutionStatus.PARTIAL_SUCCESS and len(r.errors) == 2
    assert {e["code"] for e in r.errors} == {"unsupported_command","invalid_command"}


def test_only_bad_is_failure_and_empty_is_success():
    assert execute([{"op":"wat"}]).status == ExecutionStatus.FAILURE
    assert execute([]).status == ExecutionStatus.SUCCESS


def test_block_state_and_empty_grid():
    r=execute([{"op":"set","x":0,"y":0,"z":0,"block":"oak_stairs","state":{"facing":"north"}}])
    assert r.grid.voxels[(0,0,0)].state_dict() == {"facing":"north"}
    assert VoxelGrid().bbox() is None and VoxelGrid().centroid() is None
