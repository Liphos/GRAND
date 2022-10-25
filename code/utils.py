import numpy as np
from typing import List, Tuple, Union
from scipy.spatial import KDTree



def computeNeighborsKDTree(lstPositions: Union[List[Tuple[float]], np.ndarray], distance:float=2)-> np.ndarray:
    """Create graph in O(Nlog(N))

    Args:
        lstPositions (Union[List[Tuple[float]], np.ndarray],): the positions of the antennas
        distance (float, optional): the maximum distance two antennas are considered connected. Defaults to 2.

    Returns:
        np.ndarray: _description_
    """
    lstPositions = np.array(lstPositions)
    tree_node = KDTree(lstPositions)
    pairs = tree_node.query_pairs(distance)
    return pairs

def computeNeighbors(lstPositions: Union[List[Tuple[float]], np.ndarray], n_closest:int=3) -> np.ndarray:
    """Find the closest neighbors

    Args:
        lstPositions (Union[List[Tuple[float]], np.ndarray],): the list containing the positions
        n_closest (int): The number of points that we can have for neighbors

    Returns:
        set: The edge index
    """
    closest_points = set()
    for antenna in range(len(lstPositions)):
        close_point = [np.inf for _ in range(n_closest)]
        close_point_index = [0, 0, 0]
        for antenna_close in range(len(lstPositions)):
            if antenna != antenna_close:
                dist = np.sum(np.square(list(map(lambda i, j: i - j, lstPositions[antenna], lstPositions[antenna_close]))), axis=0)
                for index in range(n_closest):
                    if dist < close_point[index]:
                        close_point[index+1:] = close_point[index:-1]
                        close_point_index[index+1:] = close_point_index[index:-1]
                                            
                        close_point[index] = dist
                        close_point_index[index] = antenna_close
                        break
        
        for antenna_close in close_point_index:
            closest_points.add((antenna, antenna_close))
    
    return closest_points
