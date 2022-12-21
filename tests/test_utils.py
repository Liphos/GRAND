"""Test utils functions"""
import numpy as np
import torch
import pytest

from core import utils

def test_compute_neighbor_kdtree(example_find_neighbors_dist):
    """Test compute_neighbor_kdtree"""
    edge_index = utils.compute_neighbor_kdtree(example_find_neighbors_dist[0])
    assert np.all(edge_index == example_find_neighbors_dist[1])

def test_compute_neighbors(example_find_neighbors_fix):
    """Test compute neighbors"""
    edges_index, dist = utils.compute_neighbors(example_find_neighbors_fix[0])
    assert np.all(edges_index == example_find_neighbors_fix[1][0])
    assert np.all(np.round(dist) == np.round(example_find_neighbors_fix[1][1]))

@pytest.mark.parametrize("pred_label, true_label, answer_mse, answer_l1", [
    (torch.tensor([0.2, 0.5, 1.2]).squeeze(-1), torch.tensor([0.1, 0.5, 1]).squeeze(-1),
     np.round((1 + 0 + 0.04)/3, 4),  np.round((1 + 0 + 0.2)/3, 4)),
])
def test_loss(pred_label, true_label, answer_mse, answer_l1):
    """Test custom losses"""
    mse_pred = utils.scaled_mse(pred_label, true_label).item()
    l1_pred = utils.scaled_l1(pred_label, true_label).item()
    assert np.round(mse_pred, 4) == answer_mse
    assert np.round(l1_pred, 4) == answer_l1
