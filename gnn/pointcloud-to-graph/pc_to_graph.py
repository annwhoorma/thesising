'''
open thesising_data_processing.ipynb to find how to: normalized pointcloud -> json description
'''

from genericpath import exists
from math import sqrt
import numpy as np
from pathlib import Path
from typing import Any
import json
import os
from tqdm import tqdm


Point = 'list[float]'
PointCloud = 'list[Point]'
Distance = float
DistanceToPoint = 'dict[Point, Distance]'
PointGraph = 'dict[Point, DistanceToPoint]'


def distance(p1: Point, p2: Point) -> float:
    assert len(p1) == len(p2) == 3
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dist = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z2 - z1) ** 2)
    return dist


def find_kcn(p: Point, pc: PointCloud, k: int) -> DistanceToPoint:
    '''finds k closest neigbours'''
    points_with_dist = map(lambda p_local: (p_local, distance(p, p_local)), pc)
    return dict(sorted(points_with_dist, key=lambda x: x[1])[1:k+1])


def distances_to_weights(k_closest: DistanceToPoint) -> DistanceToPoint:
    k_closest = dict(k_closest)
    max_dist = max(k_closest.values())
    k_closest_norm = dict(map(lambda key: (key, k_closest[key] / max_dist), k_closest))
    return k_closest_norm


def pc_to_graph(pc: PointCloud, k_neighbors: int) -> PointGraph:
    pointgraph: PointGraph = {}
    for point in pc:
        k_closest_points = find_kcn(point, pc, k_neighbors)
        k_closest_points_norm = distances_to_weights(k_closest_points)
        pointgraph[point] = k_closest_points_norm
    # point_and_dists = pointgraph[list(pointgraph.keys())[0]]
    # print(point_and_dists)
    return pointgraph


def find_edges_coo_format(pg: PointGraph, node_to_idx) -> 'tuple[Any, list[Distance]]':
    k = len(list(pg.values())[0])
    edges = np.zeros([len(pg) * k, 2])
    edges_features = []
    for i, p1 in enumerate(pg):
        neighbours = pg[p1] # dict[Point, Distance]
        p1_index = node_to_idx[p1]
        for j, (p2, p1p2_dist) in enumerate(neighbours.items()):
            p2_index = node_to_idx[p2]
            edges[i+j] = [p1_index, p2_index]
            edges_features.append(p1p2_dist)
    # print(edges.shape)
    return np.transpose(edges.astype('int32')), edges_features


def graph_to_json(pg: PointGraph, node_features, node_to_idx, idx_to_node, label, path: Path) -> Any:
    edges, edges_features = find_edges_coo_format(pg, node_to_idx)
    # print(edges.shape, len(edges_features))
    data = {
        'num_nodes': len(node_features),
        'node_features': json.dumps(node_features),
        'edges_index': json.dumps(edges.tolist()),
        'edges_attr': json.dumps(edges_features),
        'label': label
    }
    json.dump(data, open(path, 'w'))


def read_pc_from_json(json_path: Path, class_to_idx: 'dict[str, int]'):
    data = json.load(open(json_path))
    data['pc'] = json.loads(data['pc'])
    data['label'] = class_to_idx[data['label']]
    return data


def pointgraph_to_json(pc: PointCloud, k_neighbors: int, label, path_to_save):
    pc = list(map(tuple, pc))
    node_features: list[Point] = pc
    idx_to_node: dict[Point, int] = dict(enumerate(node_features))
    node_to_idx = dict([(tuple(value), key) for key, value in idx_to_node.items()])
    pg = pc_to_graph(pc, k_neighbors)
    graph_to_json(pg, node_features, node_to_idx, idx_to_node, label, path_to_save)


def read_all_pc_jsons_to_graphs(path: Path, class_to_idx: 'dict[str, int]', k_neighbours: int, path_to_save: Path) -> None:
    for f in tqdm(os.listdir(path)):
        pc = read_pc_from_json(Path(path/f), class_to_idx)
        if pc['label'] not in [0, 1, 2, 3]:
            continue
        pointgraph_to_json(pc['pc'], k_neighbours, pc['label'], path_to_save/f)


if __name__ == '__main__':
    modelnet_path = Path('ModelNet10/ModelNet10')
    folders = [dir for dir in sorted(os.listdir(modelnet_path)) if os.path.isdir(modelnet_path/dir)]
    class_to_idx = {folder: i for i, folder in enumerate(folders)}

    # k_neighbours = 25
    # Path('modelnet_graphs_4_classes').mkdir(exist_ok=True)
    # Path('modelnet_graphs_4_classes/train').mkdir(exist_ok=True)
    # Path('modelnet_graphs_4_classes/test').mkdir(exist_ok=True)
    # read_all_pc_jsons_to_graphs(Path('modelnet_jsons/train'), class_to_idx, k_neighbours, Path('modelnet_graphs_4_classes/train'))
    # read_all_pc_jsons_to_graphs(Path('modelnet_jsons/test'), class_to_idx, k_neighbours, Path('modelnet_graphs_4_classes/test'))
    # read_all_pc_jsons_to_graphs(Path('just_one_graph/'), class_to_idx,  k_neighbours, Path('just_one_graph_pc'))