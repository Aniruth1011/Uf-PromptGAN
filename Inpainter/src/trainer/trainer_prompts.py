import os
import importlib
from tqdm import tqdm
from glob import glob

import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
#from model.aotgan import InpaintGenerator #, Discriminator 
from model.aotganwithprompts import InpaintGenerator , Discriminator
from data import create_loader 
from loss import loss as loss_module
from .common import timer, reduce_loss_dict
import torch.nn as nn 

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def initialize_weights(layer, range=(-0.1, 0.1)):
  if hasattr(layer, 'weight'):
    nn.init.uniform_(layer.weight, range[0], range[1])


class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0

        # setup data set and data loader
        self.dataloader = create_loader(args)

        # set up losses and metrics
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        self.adv_loss = getattr(loss_module, args.gan_type)()

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]
        #net = importlib.import_module('model.'+args.model) i changed 

        net1 = InpaintGenerator(args).cuda()
        
        self.netG = net1 #net.InpaintGenerator(args).cuda()
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))
        
        prompt_layers = ["middle.0.globalprompt.prompt_param", "middle.0.globalprompt.linear_layer.weight", "middle.0.globalprompt.linear_layer.bias", "middle.0.globalprompt.conv3x3.weight", "middle.1.globalprompt.prompt_param", "middle.1.globalprompt.linear_layer.weight", "middle.1.globalprompt.linear_layer.bias", "middle.1.globalprompt.conv3x3.weight", "middle.2.globalprompt.prompt_param", "middle.2.globalprompt.linear_layer.weight", "middle.2.globalprompt.linear_layer.bias", "middle.2.globalprompt.conv3x3.weight", "middle.3.globalprompt.prompt_param", "middle.3.globalprompt.linear_layer.weight", "middle.3.globalprompt.linear_layer.bias", "middle.3.globalprompt.conv3x3.weight", "middle.4.globalprompt.prompt_param", "middle.4.globalprompt.linear_layer.weight", "middle.4.globalprompt.linear_layer.bias", "middle.4.globalprompt.conv3x3.weight", "middle.5.globalprompt.prompt_param", "middle.5.globalprompt.linear_layer.weight", "middle.5.globalprompt.linear_layer.bias", "middle.5.globalprompt.conv3x3.weight", "middle.6.globalprompt.prompt_param", "middle.6.globalprompt.linear_layer.weight", "middle.6.globalprompt.linear_layer.bias", "middle.6.globalprompt.conv3x3.weight", "middle.7.globalprompt.prompt_param", "middle.7.globalprompt.linear_layer.weight", "middle.7.globalprompt.linear_layer.bias", "middle.7.globalprompt.conv3x3.weight"]
        pretrained = torch.load(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/experiments/aotgan_flareremovaltrainwithprompts_b8_pconv256/G0090000.pt' , map_location='cuda')
        for name, param in net1.named_parameters():
            if name in prompt_layers:
                initialize_weights(param)
            else:
                if hasattr(name, 'weight'):
                    net1.name.weight = pretrained.name.weight

        #net1 = InpaintGenerator(args).cuda()
        #net1.load_state_dict(torch.load(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/experiments/aotgan_places2_pconv512/G0000000.pt' , map_location='cuda'))
    
        net2 = Discriminator().cuda()
        self.netD = net2 #net.Discriminator().cuda()
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))
        net2.load_state_dict(torch.load(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/experiments/aotgan_places2_pconv512/D0000000.pt' ,  map_location='cuda'))

        
        self.load()
        if args.distributed:
            self.netG = DDP(self.netG, device_ids= [args.local_rank], output_device=[args.local_rank])
            self.netD = DDP(self.netD, device_ids= [args.local_rank], output_device=[args.local_rank])
        
        if args.tensorboard: 
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
            

    def load(self):
        try: 
            gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'G*.pt'))))[-1]
            self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
            self.netG.load_state_dict(torch.load(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/experiments/aotgan_flareremovaltrainwithprompts_b8_pconv256/G0090000.pt' , map_location='cuda'))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            if self.args.global_rank == 0: 
                print(f'[**] Loading generator network from {gpath}')
        except: 
            pass 
        
        try: 
            dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'D*.pt'))))[-1]
            self.net2.load_state_dict(torch.load(r'/home/santhi/MIPI_Promptir/MIPI/AOT-GAN-for-Inpainting/experiments/aotgan_places2_pconv512/D0000000.pt' ,  map_location='cuda'))
            if self.args.global_rank == 0: 
                print(f'[**] Loading discriminator network from {dpath}')
        except: 
            pass
        
        try: 
            opath = sorted(list(glob(os.path.join(self.args.save_dir, 'O*.pt'))))[-1]
            data = torch.load(opath, map_location='cuda')
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            if self.args.global_rank == 0: 
                print(f'[**] Loading optimizer from {opath}')
        except: 
            pass
 

    def save(self, ):
        if self.args.global_rank == 0:
            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
            torch.save(self.netG.state_dict(), 
                os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
            # torch.save(self.netD.state_dict(), 
            #     os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            # torch.save(
            #     {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()}, 
            #     os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))
            

    def train(self):
        pbar = range(self.iteration, self.args.iterations)
        if self.args.global_rank == 0: 
            pbar = tqdm(range(self.args.iterations), initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
            timer_data, timer_model = timer(), timer()
        
        for idx in tqdm(pbar):
            self.iteration += 1
            images, masks, filename = next(self.dataloader)
            images, masks = images.cuda(), masks.cuda()
            images_masked = (images * (1 - masks).float()) + masks

            if self.args.global_rank == 0: 
                timer_data.hold()
                timer_model.tic()

            # in: [rgb(3) + edge(1)]
            pred_img = self.netG(images_masked, masks)
            comp_img = (1 - masks) * images + masks * pred_img

            # reconstruction losses 
            losses = {}
            for name, weight in self.args.rec_loss.items(): 
                losses[name] = weight * self.rec_loss_func[name](pred_img, images)
            
            # adversarial loss 
            dis_loss, gen_loss = self.adv_loss(self.netD, comp_img, images, masks)
            losses[f"advg"] = gen_loss * self.args.adv_weight
            
            # backforward 
            self.optimG.zero_grad()
            self.optimD.zero_grad()
            sum(losses.values()).backward()
            losses[f"advd"] = dis_loss 
            dis_loss.backward()
            self.optimG.step()
            self.optimD.step()

            if self.args.global_rank == 0:
                timer_model.hold()
                timer_data.tic()

            # logs
            scalar_reduced = reduce_loss_dict(losses, self.args.world_size)
            if self.args.global_rank == 0 and (self.iteration % self.args.print_every == 0): 
                pbar.update(self.args.print_every)
                description = f'mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                for key, val in losses.items(): 
                    description += f'{key}:{val.item():.3f}, '
                    if self.args.tensorboard: 
                        self.writer.add_scalar(key, val.item(), self.iteration)
                pbar.set_description((description))
                if self.args.tensorboard: 
                    self.writer.add_image('mask', make_grid(masks), self.iteration)
                    self.writer.add_image('orig', make_grid((images+1.0)/2.0), self.iteration)
                    self.writer.add_image('pred', make_grid((pred_img+1.0)/2.0), self.iteration)
                    self.writer.add_image('comp', make_grid((comp_img+1.0)/2.0), self.iteration)
                    
            
            if self.args.global_rank == 0 and (self.iteration % self.args.save_every) == 0: 
                self.save()
