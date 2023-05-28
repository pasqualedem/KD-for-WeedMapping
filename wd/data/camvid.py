import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.pytorch.functional import img_to_tensor
import cv2 as cv
from PIL import Image
from pathlib import Path

from os.path import join as pjoin
from ezdl.data import DatasetInterface
from ezdl.transforms import Denormalize

class CamVidDatasetInterface(DatasetInterface):
    size = (3, 720, 960)
    lib_dataset_params = {
            'mean': [0.39068785, 0.40521392, 0.41434407],
            'std': [0.29652068, 0.30514979, 0.30080369]
        }

    def __init__(self, dataset_params={}):
        super().__init__(dataset_params)
        self.root_dir = dataset_params['root']


        train_images = list(Path(pjoin(self.root_dir, 'train')).glob('*'))
        train_labels = list(Path(pjoin(self.root_dir, 'train_labels')).glob('*'))
        train_images.sort()
        train_labels.sort()
        
        classes = pd.read_csv(pjoin(self.root_dir, "class_dict.csv"))
        labels = dict(zip(range(len(classes)), classes['name'].tolist()))
        color_map = dict(zip(range(len(classes)),classes.iloc[:,1:].values.tolist()))
        self.cmap = color_map
        
        assert len(train_images) == len(train_labels)
        
        train_set = list(zip(train_images,train_labels))
        
        val_images = list(Path(pjoin(self.root_dir, 'val')).glob('*'))
        val_labels = list(Path(pjoin(self.root_dir, 'val_labels')).glob('*'))
        val_images.sort()
        val_labels.sort()
        
        assert len(val_images) == len(val_labels)
        
        val_set = list(zip(val_images,val_labels))
        
        test_images = list(Path(pjoin(self.root_dir, 'test')).glob('*'))
        test_labels = list(Path(pjoin(self.root_dir, 'test_labels')).glob('*'))
        test_images.sort()
        test_labels.sort()
        
        assert len(test_images) == len(test_labels)
        
        test_set = list(zip(test_images,test_labels))

        train_transform, test_transform = self.get_transforms(dataset_params)

        self.trainset = CamVidDataset(train_set, color_map, labels, train_transform, return_name=dataset_params.get('return_name', False))
        self.valset = CamVidDataset(val_set, color_map, labels, test_transform, return_name=dataset_params.get('return_name', False))
        self.testset = CamVidDataset(test_set, color_map, labels, test_transform, return_name=dataset_params.get('return_name', False))

    def get_transforms(self, dataset_params={}):
        size = dataset_params.get('size', self.size[1:])

        test_transform = A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])
        transform = A.Compose([
            A.ColorJitter(),
            A.HorizontalFlip(),
            A.Resize(size[0], size[1]),
            A.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])

        return transform, test_transform
    
    def undo_preprocess(self, x):
        return (Denormalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std'])(x) * 255).type(torch.uint8)


class CamVidDataset:
    def __init__(self, split, cmap, labels, transforms, single=None, return_name=False):
        self.split = split
        self.cmap = cmap
        self.transforms = transforms
        self.classes = {y:x for x,y in labels.items()}
        self.CLASS_LABELS = labels
        self.return_name = return_name
        self.classes = list(labels.keys())

        self.single = self.classes[single] if single is not None else None
        
        
    def __len__(self):
        return len(self.split)
    
    def __getitem__(self,idx):
        
        sample,seg = self.split[idx]
        
        sample = cv.imread(str(sample))
        sample = cv.cvtColor(sample,cv.COLOR_BGR2RGB)
        
        seg = cv.imread(str(seg))
        seg = cv.cvtColor(seg,cv.COLOR_BGR2RGB)
        
        aug = self.transforms(image=sample,mask=seg)
            
        sample,seg = aug['image'], aug['mask']
        
        mask_shape = *seg.shape[:2],len(self.cmap)
        mask = np.zeros(mask_shape)
        
        for i,color in self.cmap.items():
            mask[:,:,i] = np.all(np.equal(seg,color),axis=-1)
            
            
        if self.single is not None:
            mask = mask[:,:,self.single]
            mask = torch.tensor(mask, dtype=torch.long).unsqueeze(2).permute(2,0,1).argmax(0)
        else:
            mask = torch.tensor(mask, dtype=torch.long).permute(2,0,1).argmax(0)
            
        sample = img_to_tensor(sample)
        if self.return_name:
            return sample, mask, {'input_name': str(self.split[idx][0]).split("/")[-1]}

        return sample, mask