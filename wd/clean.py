import argparse
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from wd.data.weedmap import WeedMapDataset
from tqdm import tqdm

import torchvision.transforms as transforms
import os
import shutil


problematic_images = {
    '007': [
      22,
      23,
      24,
      32,
      33,
      42,
      43,
      44,
      45,
      55,
      56,
      66,
      67,
      68,
      69,
      80,
      81,
      82,
      91,
      92,
      93,
      94,
      95,
      96,
      104,
      105,
      106,
      108,
      109,
      110,
      116,
      117,
      118,
      120,
      121,
      122,
      123,
      127,
      128,
      129,
      130,
      131,
      132,
      133,
      134,
      135,
      138,
      139,
      140,
      141,
      142,
      143,
      144,
      145,
      146,
      147,
      148,
      149,
      150,
      151,
      152,
      153,
      154,
      155,
      156,
      157,
      158,
      159,
      160,
      161,
      162,
      163,
      164,
      165,
      166,
      167,
      168,
      169,
      170,
      171,
      172,
      173,
      174,
      175,
      176,
      177,
      178,
      179,
      1,
      2,
      6,
      7,
      8,
      9,
      13,
      14,
      15,
      16,
      0
  ],
  '005': [
    2,
    6,
    37,
    45,
    182,
    195,
    198,
    210,
    215,
    218,
    219,
    220,
    223
  ],

  '006': [
    19,
    22,
    23,
    31,
    115,
    127,
    139,
    140
  ]
}


INPATH = 'dataset/processed'
OUTPATH = "dataset/cleaned"

SEQUOIA_CHANNELS = ['CIR', 'G', 'NDVI', 'NIR', 'R', 'RE']
REDEDGE_CHANNELS = ['CIR', 'G', 'NDVI', 'NIR', 'R', 'RE', 'B']


def get_first_element_indices(lst):
    prev = lst[0][0]
    indices = [0]
    for i in range(len(lst)):
        if lst[i][0] != prev:
            prev = lst[i][0]
            indices.append(i)
    return indices    


def delete_problematic_imgs(root, channels, tempdir_check=None):
    if tempdir_check:
        shutil.rmtree(tempdir_check, ignore_errors=True)
        os.makedirs(tempdir_check, exist_ok=True)
    trs = transforms.Compose([])
    dataset = WeedMapDataset(root, transform=trs, return_mask=True, target_transform=trs)
    counter = 0
    dataset.index.sort()
    starts = get_first_element_indices(dataset.index)
    folders = sorted({folder for folder, _ in dataset.index})
    starts = {folder: starts[i] for i, folder in enumerate(folders)}
    for i in tqdm(range(len(dataset))):
        folder, img_name = dataset.index[i]
        img, (gt, mask) = dataset[i]
        j = i - starts[folder]
        if j in problematic_images[folder]:
            gt.close()
            mask.close()
            gt_path_color = os.path.join(root, folder, 'groundtruth',
                                         folder + '_' + img_name.split('.')[0] + '_GroundTruth_color.png'
                                         )
            gt_path_imap = os.path.join(root, folder, 'groundtruth',
                                        folder + '_' + img_name.split('.')[0] + '_GroundTruth_iMap.png'
                                        )
            gt_path = os.path.join(root, folder, 'groundtruth',
                                   folder + '_' + img_name)
            mask_path = os.path.join(root, folder, 'mask', img_name)

            if tempdir_check:
                if isinstance(img, torch.Tensor):
                    img = Image.fromarray(img.byte().permute(1, 2, 0).numpy())
                img.save(os.path.join(tempdir_check, img_name))
                shutil.copy(gt_path_color, os.path.join(tempdir_check,
                                                        folder + '_' + img_name.split('.')[0] + '_GroundTruth_color.png'))
            os.remove(gt_path_color)
            os.remove(gt_path_imap)
            os.remove(gt_path)
            os.remove(mask_path)
            for c in channels:
                c_path = os.path.join(root, folder, 'tile', c, img_name)
                os.remove(c_path)
            counter += 1
    print('Removed {} problematic images'.format(counter))


def copy_dataset(inpath, outpath):
    shutil.copytree(inpath, outpath, dirs_exist_ok=True)
    print('Dataset copied')


def clean(subset):
    if subset == "Sequoia":
        # SEQUOIA 139 removed
        channels = SEQUOIA_CHANNELS
    elif subset == "RedEdge":
        # REDEDGE 413 removed
        channels = REDEDGE_CHANNELS
    else:
        raise NotImplementedError("Not valid subset")

    os.makedirs(os.path.join(OUTPATH, subset), exist_ok=True)
    copy_dataset(os.path.join(INPATH, subset), os.path.join(OUTPATH, subset))
    delete_problematic_imgs(os.path.join(OUTPATH, subset), channels, tempdir_check='tmp')



