from offline_evaluation.server_v2.statistics import paired_cluster_ci, holm_adjust

def test_buildings_equal_weight_and_failures_not_zero():
    rows = [{'scene_id': 'a', 'a': 0, 'b': 1}] * 10 + [{'scene_id': 'b', 'a': 0, 'b': 0},
            {'scene_id': 'c', 'a': 1, 'b': 0, 'status_b': 'execution_failure'}]
    x = paired_cluster_ci(rows, 'iou', replicates=100)
    assert x['mean_delta'] == .5 and x['buildings'] == 2 and x['excluded_pairs'] == 1
    assert x == paired_cluster_ci(rows, 'iou', replicates=100)

def test_holm_known_values():
    assert holm_adjust([.01, .04, .2]) == [.03, .08, .2]
