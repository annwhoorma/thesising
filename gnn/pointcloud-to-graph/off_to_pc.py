import json
from pathlib import Path
from ...helping import read_off
import os
from torchvision import transforms


# TOFIX: folder=train/test should be enum
class PointCloudJSON:
    def __init__(self, root_dir, static_transform, folder="train", folders=None):
        if not folders:
            folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.pcs = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    with open(new_dir/file, 'r') as f:
                        verts, faces = read_off(f)
                    sample['pc'] = static_transform((verts, faces))
                    sample['label'] = category
                    self.pcs.append(sample)

def export_to_jsons(pcs, path: Path):
    for i, sample in enumerate(pcs):
        sample = {
            'pc': json.dumps(sample['pc'].tolist()),
            'label': sample['label']
        }
        json.dump(sample, open(f'{path}/{i}.json', 'w'))


Path('pc_jsons').mkdir(exist_ok=True)
Path('pc_jsons/train').mkdir(exist_ok=True)
Path('pc_jsons/test').mkdir(exist_ok=True)

static_trs = transforms.Compose([
                               PointSampler(1024),
                               Normalize(),
])

pcjson_train = PointCloudJSON(Path('ModelNet10'), static_trs, 'train') # TOFIX
export_to_jsons(pcjson_train.pcs, Path('pc_jsons/train'))

pcjson_test = PointCloudJSON(Path('ModelNet10'), static_trs, 'test') # TOFIX
export_to_jsons(pcjson_test.pcs, Path('pc_jsons/test'))