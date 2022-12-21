"""Conftest for the utils"""
import pytest
import numpy as np
import torch

@pytest.fixture
def example_find_neighbors_dist():
    """Example of graph"""
    graph_nodes = np.array([[0, 1], [0,2], [1,2], [-1, -1], [-2, -1], [0, 4]])*1000
    solution = np.array([[0, 1], [0, 2], [1, 2], [3, 4]])
    return (graph_nodes, solution)

@pytest.fixture
def example_find_neighbors_fix():
    """Example of graph"""
    graph_nodes = np.array([[0, 1], [0,2], [1,2], [-1, -1], [0, 4]])*1000
    solution = np.array([[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 4], [2, 1], [2, 0], [2, 4],
                         [3, 0], [3, 1], [3, 2], [4, 1], [4, 2], [4, 0]])
    distances = 1000 * np.array([1, np.sqrt(2), np.sqrt(5), 1, 1, 2, 1, np.sqrt(2), np.sqrt(5),
                          np.sqrt(5), np.sqrt(10), np.sqrt(13), 2, np.sqrt(5), 3])
    return (graph_nodes, (solution, distances))

