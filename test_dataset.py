import os
import cv2
import numpy as np
import torch
#import imgcrop
import random
import math
from PIL import Image, ImageDraw

from torchvision import transforms
from torch.utils.data import Dataset

import utils
import pickle as pkl


class InpaintDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = sorted(utils.get_files(opt.baseroot), ) # key=lambda d:int(d.split('/')[-1].split('.')[0])
        self.masklist = sorted(utils.get_files(opt.baseroot_mask), ) # key=lambda d:int(d.split('/')[-1].split('.')[0])

        self.pkldatalist = sorted(utils.get_files("./test_param"), )

        self.imgsize = (opt.imgsize, opt.imgsize )

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        shape = self.imgsize # (256, 256)
        # image
        img = cv2.imread(self.imglist[index])
        img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)

        mask = cv2.imread(self.masklist[index])[:, :, 0]
        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_AREA)

        # find the Minimum bounding rectangle in the mask
        '''
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cidx, cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            mask[y:y+h, x:x+w] = 255
        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()


        with open(self.pkldatalist[index], 'rb') as file:
            pkldata = pkl.load(file )
        verts = torch.from_numpy(pkldata['vertices']).contiguous() # .unsqueeze(0)

        return img, mask, verts
