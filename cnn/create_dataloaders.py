from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from helping import *
import numpy as np

def find_weights_for_balanced_classes(pcs, nclasses):
    count_classes = np.zeros(nclasses)
    for pc in pcs:
        count_classes[pc[1]] += 1
    num_pcs = np.sum(count_classes)
    count_classes = list(map(lambda x: num_pcs / x, count_classes))
    return list(map(lambda pc: count_classes[pc[1]], pcs))


static_trs = transforms.Compose([
                               PointSampler(1024),
                               Normalize(),
])

dynamic_trs = transforms.Compose([
                            #    RandomRotation(),
                               RandomNoise(),
                               ToSorted(),
                               ToTensor()
])


path = Path('ModelNet10')
folders = ['bathtub', 'chair', 'desk']
num_classes = len(folders)
bs = 32

train_dataset = PointCloudDataBoth(path, folders=folders, static_transform=static_trs, later_transform=dynamic_trs)
train_dataset_weights = find_weights_for_balanced_classes(train_dataset, num_classes)
print(train_dataset_weights)
train_weights = torch.DoubleTensor(train_dataset_weights)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=bs, drop_last=True)

valid_dataset = PointCloudDataBoth(path, folder='test', folders=folders, static_transform=static_trs)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs, drop_last=True)

Path('dataloaders').mkdir(exist_ok=True)
torch.save(train_loader, 'dataloaders/trainloader.pth')
torch.save(valid_loader, 'dataloaders/validloader.pth')
