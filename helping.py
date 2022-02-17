import numpy as np
import math
import random
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from pathlib import Path

def default_transforms():
    return transforms.Compose([
                               PointSampler(1024),
                               Normalize(),
                               RandomNoise(),
                               ToSorted(),
                               ToTensor()
    ])


def read_off(file):
    if file.readline().strip() != 'OFF':
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple(
        int(s) for s in file.readline().strip().split(' ')
    )

    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig


def pcshow(xs, ys, zs):
    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=3,
                                  line=dict(width=2,
                                            color='Black')),
                      selector=dict(mode='markers'))
    fig.show()



class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                      weights=areas,
                                      cum_weights=None,
                                      k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))
        
        return sampled_points

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        return pointcloud + noise

class RandomRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])

        return rot_matrix.dot(pointcloud.T).T

class ToSorted(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        return np.array(sorted(pointcloud.tolist()))

class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms(), folders=None):
        self.root_dir = root_dir
        if not folders:
            folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.valid = valid
        self.pcs = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    with open(new_dir/file, 'r') as f:
                        verts, faces = read_off(f)
                    sample['pc'] = (verts, faces)
                    sample['category'] = category
                    self.pcs.append(sample)

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, idx):
        pointcloud = self.transforms(self.pcs[idx]['pc'])
        category = self.pcs[idx]['category']
        return pointcloud, self.classes[category]
    
class PointCloudDataPre(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms(), folders=None):
        self.root_dir = root_dir
        if not folders:
            folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.valid = valid
        self.pcs = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    with open(new_dir/file, 'r') as f:
                        verts, faces = read_off(f)
                    sample['pc'] = self.transforms((verts, faces))
                    sample['category'] = category
                    self.pcs.append(sample)

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, idx):
        pointcloud = self.pcs[idx]['pc']
        category = self.pcs[idx]['category']
        return pointcloud, self.classes[category]
    
    
class PointCloudDataBoth(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", static_transform=default_transforms(), later_transform=None, folders=None):
        self.root_dir = root_dir
        if not folders:
            folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.static_transform = static_transform
        self.later_transform = later_transform
        self.valid = valid
        self.pcs = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    with open(new_dir/file, 'r') as f:
                        verts, faces = read_off(f)
                    sample['pc'] = self.static_transform((verts, faces))
                    sample['category'] = category
                    self.pcs.append(sample)

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, idx):
        pointcloud = self.pcs[idx]['pc']
        if self.later_transform is not None:
            pointcloud = self.later_transform(pointcloud)
        category = self.pcs[idx]['category']
        return pointcloud, self.classes[category]
