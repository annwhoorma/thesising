import json
from pathlib import Path
from helping import *
import os
from torchvision import transforms
from tqdm import tqdm


# TOFIX: folder=train/test should be enum
class PointCloudJSON:
    def __init__(self, root_dir, static_transform, path_to_save, folder="train", folders=None):
        if not folders:
            folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        for category in self.classes.keys():
            self.pcs = []
            new_dir = root_dir/Path(category)
            for file in tqdm(os.listdir(new_dir), desc=category):
                if file.endswith('.off'):
                    with open(new_dir/file, 'r') as f:
                        verts, faces = read_off(f)
                    pc = static_transform((verts, faces))
                    self.pcs.append(pc)
            export_to_jsons(self.pcs, label=category, path=path_to_save)

def export_to_jsons(pcs, label, path: Path):
    Path(path/label).mkdir(exist_ok=True)
    for i, pc in enumerate(pcs):
        sample = {
            'pc': json.dumps(pc.tolist()),
            'label': label
        }
        json.dump(sample, open(f'{path}/{label}/{i}.json', 'w'))


# Path('pc_jsons_n').mkdir(exist_ok=True)
# Path('pc_jsons_n/train').mkdir(exist_ok=True)

static_trs = transforms.Compose([
                               PointSampler(1024),
                               Normalize(),
])

path = '../../datasets/ModelNet10'
pcjson_train = PointCloudJSON(Path(path), static_trs, Path('pc_jsons/train'), 'train') # TOFIX  folder=train/test should be enum