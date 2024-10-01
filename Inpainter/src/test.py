
import os
import argparse
import importlib
import numpy as np
from PIL import Image
from glob import glob
import cv2 

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision import transforms 

from model.aotgan import InpaintGenerator
#from model.aotganwithprompts import InpaintGenerator
from utils.option import args 

from natsort import natsorted

#from datasets import Image
from PIL import Image 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)
 

def main_worker(args, use_gpu=True):  

    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    
    # Model and version
    #net = importlib.import_module('model.'+args.model)
    
    model = InpaintGenerator(args).cuda() #net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/experiments/aotgan_places2_pconv512/G0000000.pt', map_location='cuda'))
    model.eval()
    print("loaded model")
    # prepare datasets
    image_paths = []
    
    """
    for ext in ['.jpg', '.png']: 
        image_paths.extend(glob(os.path.join(args.dir_image, '*'+ext)))
    image_paths.sort()"""
    print(1)
    image_paths = natsorted([os.path.join(r'/home/santhi/MIPI_Promptir/MIPI/Submission/Uformer/deflare' , i) for i in os.listdir(r'/home/santhi/MIPI_Promptir/MIPI/Submission/Uformer/deflare')])
    mask_paths = natsorted([os.path.join(r'/home/santhi/MIPI_Promptir/MIPI/Submission/Mask' , i) for i in os.listdir(r'/home/santhi/MIPI_Promptir/MIPI/Submission/Mask')])
    print(2)
    #image_paths = r'/home/santhi/MIPI_Promptir/MIPI/Uformer_outputs/test_/img_resc/000142.png'
    #mask_paths = r'/home/santhi/MIPI_Promptir/MIPI/Masks/test_/000142.png'

    #mask_paths = sorted(glob(os.path.join(args.dir_mask, '*.png')))
    os.makedirs(args.outputs, exist_ok=True)

    # iteration through datasets
    print(len(image_paths))
    for ipath, mpath in zip(image_paths, mask_paths): 

        image = resizer(ToTensor()(Image.open(ipath).convert('RGB')))
        image = (image * 2.0 - 1.0).unsqueeze(0)
        mask = ToTensor()(Image.open(mpath).convert('L'))
        mask = mask.unsqueeze(0)

        print(image.shape , mask.shape)

        image, mask = image.cuda(), mask.cuda()

        print(image.shape)

        #print(image.shape)
        #print(mask.shape)
        #print(image.shape[1] )
        #print((image.shape[-2], image.shape[-1]))
        #resizer = transforms.Resize((image.shape[-2], image.shape[-1]))
        # resizer_ = transforms.Resize((1920 , 887))  # Set interpolation to bilinear (2)
        image_masked = image * (1 - (mask)).float() + (mask)
        print(image_masked.shape , mask.shape) 
        with torch.no_grad():
            pred_img = model(image_masked, mask)

        # if mask.shape != torch.Size([1 , 1  , 1920 , 1440]) :
        #     if mask.shape != torch.Size([1 , 1  , 1920 , 887]) or image_masked.shape != torch.Size([1, 3, 1920, 887]) or image.shape != torch.Size([1, 3, 1920, 887]):
        #         resizer_ = transforms.Resize((1920 , 887))
        #         mask_ = resizer_(mask)
        #         image_masked_ = resizer_(image_masked)
        #         with torch.no_grad():
        #             pred_img_ = model(image_masked_, mask_)
        #         comp_imgs = (1 - mask_) * image_masked_ + mask_ * pred_img_
        #         # image_name = os.path.basename(ipath).split('.')[0]
        # else:
        comp_imgs = (1 - mask) * image_masked + mask * pred_img
    
        image_name = os.path.basename(ipath).split('.')[0]

        # postprocess(image_masked[0]).save(os.path.join(args.outputs, f'{image_name}_masked.png'))
        # postprocess(pred_img[0]).save(os.path.join(args.outputs, f'{image_name}_pred.png'))
        # postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f'{image_name}_comp.png'))
        postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f'{image_name}.png'))
        print(f'saving to {os.path.join(args.outputs, image_name)}')

if __name__ == '__main__':
    main_worker(args)
