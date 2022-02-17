from pathlib import Path
import os
import json
import numpy as np

def read_dataset(path):
    jsons = os.listdir(Path(path))
    shapes = set()
    for js in jsons:
        graph_path = Path(path, js)
        data = json.load(open(graph_path))

        num_nodes = data['num_nodes']

        edge_index = np.array(json.loads(data['edges_index']))
        edge_attr = np.array(json.loads(data['edges_attr']))
        shapes.add(edge_index.shape)
    print(shapes)


read_dataset(Path('modelnet_graphs/train'))