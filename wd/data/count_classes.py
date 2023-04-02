import multiprocessing

import torch
import torch.nn.functional as F
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import transforms

from ezdl.transforms import SegOneHot, ToLong, FixValue, squeeze0

from wd.data.weedmap import WeedMapDataset

WORKERS = multiprocessing.cpu_count()
BATCH_SIZE = 256
WIDTH = 360
HEIGHT = 480
CLASSES = 3


def count(root, folders, channels):
    """
    Calculate the mean and the standard deviation of a dataset
    """
    target_transform = transforms.Compose([
            transforms.PILToTensor(),
            squeeze0,
            ToLong(),
            FixValue(source=10000, target=1),
            SegOneHot(num_classes=len(WeedMapDataset.CLASS_LABELS.keys()))
        ]) 

    index = WeedMapDataset.build_index(
        root,
        macro_folders=folders,
        channels=channels,
    )

    sq = WeedMapDataset(root,
                        transform=lambda x: x,
                        target_transform=target_transform,
                        index=index,
                        channels=channels
                        )


    # data loader
    image_loader = DataLoader(sq,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    # placeholders
    count = torch.zeros(CLASSES)

    # loop through images
    for _, target in tqdm(image_loader):
        count += target.sum(axis=[0, 2, 3])

    print(count)
    rel = count / count.sum()
    print(rel)
    print(1 / (rel / rel[1]))



def count_classes():
    # SEQUOIA
    root = "./dataset/4_rotations_cleaned_006_test/Sequoia"
    folders = ['005', '007']
    channels = ['R', 'G', 'NDVI', 'NIR', 'RE']
    # 005 007 [0.0062, 1.0000, 1.9566] processed
    # 005 007 [0.0074, 1.0000, 2.1376] cleaned

    # REDEDGE
    # root = "./dataset/processed/RedEdge"
    # folders = ['000', '001', '002', '004']
    # channels = ['R', 'G', 'B', 'NDVI', 'NIR', 'RE']
    # d = {}
    # for folder in folders:
    #     d[folder] = calculate(root, [folder], channels)
    
    count(root, folders, channels)