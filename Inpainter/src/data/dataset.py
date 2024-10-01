import os
import math
import numpy as np
from glob import glob

from random import shuffle
from PIL import Image, ImageFilter

from natsort import natsorted

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type

        self.image_path = []

        print("Dir" , args.dir_image)
        print("data_Train" , args.data_train)
        """
        for ext in ['*.jpg', '*.png']: 
            self.image_path.extend(glob(os.path.join(args.dir_image, args.data_train, ext)))
            self.mask_path = glob(os.path.join(args.dir_mask, args.mask_type, '*.png'))"""
        
        self.images = natsorted(os.listdir(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/data/image/deflare_resized'))
        self.image_path = [os.path.join(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/data/image/deflare_resized' , i ) for i in self.images]

        self.masks= natsorted(os.listdir(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/data/mask'))
        self.mask_path = [os.path.join(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/data/mask' , i) for i in self.masks]

        print("Len from loader" , len(self.image_path))

            # augmentation 
        min_val , max_val = [-1 , 1]
        mean = [0.5, 0.5, 0.5]  # Assuming image channels are R, G, B in range [0, 1]
        std = [0.5, 0.5, 0.5]   # Standard deviation for each channel
        self.img_trans = transforms.Compose([
                transforms.RandomResizedCrop((args.image_size  , args.image_size ))  , #, i removed this 
                ##transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST) , #i put this 
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor() , 
                                transforms.Normalize(mean=mean, std=std) , 
])
        mean = [0.5]
        std = [0.5]
        self.mask_trans = transforms.Compose([
            #transforms.Normalize(mean=mean, std=std) ,
    transforms.Resize((256  , 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((0, 45), interpolation=transforms.InterpolationMode.NEAREST),
])


        #print("Len from dataloader" , len(self.image_path))

            
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
            # load image
        image = Image.open(self.image_path[index]).convert('RGB')
        filename = os.path.basename(self.image_path[index])

        if self.mask_type == 'pconv':
            index = np.random.randint(0, len(self.mask_path))
            mask = Image.open(self.mask_path[index])
            mask = mask.convert('L')
        else:
            mask = np.zeros((self.h, self.w)).astype(np.uint8)
            mask[self.h//4:self.h//4*3, self.w//4:self.w//4*3] = 1
            mask = Image.fromarray(mask).convert('L')
            
            # augment
        image = self.img_trans(image) * 2. - 1.
        mask = F.to_tensor(self.mask_trans(mask))

        # print("Image" , image.shape)
        # print("mask" , mask.shape)

        return image, mask, filename



if __name__ == '__main__': 

    from attrdict import AttrDict
    args = {
        'dir_image': '../Datasets/image/',
        'data_train': 'places2',
        'dir_mask': '../Datasets/mask',
        'mask_type': 'pconv',
        'image_size': 512
    }
    args = AttrDict(args)

    data = InpaintingData(args)
    print(len(data), len(data.mask_path))
    img, mask, filename = data[0]
    print(img.size(), mask.size(), filename)
