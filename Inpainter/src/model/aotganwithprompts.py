import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .common import BaseNetwork

class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=256,prompt_len=4,prompt_size = 96,lin_dim = 256):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt

class InpaintGenerator(BaseNetwork):
    def __init__(self, args, old_model=None):  # Pass old_model for weight transfer (optional)
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[self._load_or_create_aot_block(256, args.rates, old_model) for _ in range(args.block_num)])
        #self.middle = nn.Sequential(*[self._load_or_create_aot_block(512, args.rates, old_model) for _ in range(args.block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.init_weights()  # Initialize weights for encoder, decoder, and newly created AOTBlocks

    def _load_or_create_aot_block(self, dim, rates, old_model):
        """Loads weights from old AOTBlock or creates a new one with random weights."""
        if old_model is not None:
            try:
                # Attempt to load weights from the old AOTBlock (assuming first in the sequence)
                old_block = old_model.middle[0]
                new_block = AOTBlock(dim, rates)

                # Copy weights for shared blocks (`block{i}`) explicitly
                for i, rate in enumerate(rates):
                    new_block.__setattr__(f'block{str(i).zfill(2)}',
                                         old_block.__getattr__(f'block{str(i).zfill(2)}').clone())  # Clone weights

                # Load weights for fuse and gate layers
                new_block.fuse.load_state_dict(old_block.fuse.state_dict())
                new_block.gate.load_state_dict(old_block.gate.state_dict())
                return new_block
            except:
                pass  # If loading fails, continue to create a new AOTBlock with random weights

        # If no old model provided or loading fails, create a new AOTBlock with random weights
        return AOTBlock(dim, rates)

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
            #nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.globalprompt = PromptGenBlock()

    def forward(self, x):
        global_prompt = self.globalprompt(x)
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]

        out = torch.cat(out,  1)
        """#print(out.shape  , global_prompt.shape)
        #print(out.shape)
        print(out.shape , global_prompt.shape)

        out_fl = out.view(-1, 64*64, 256).permute(1, 0, 2)
        glob_fl = global_prompt.view(-1, 64*64, 256).permute(1, 0, 2)

        attention = nn.MultiheadAttention(embed_dim=256, num_heads=8).cuda()

        attn_output, _ = attention(out_fl, glob_fl, glob_fl)

        attn_output = attn_output.permute(1, 0, 2).reshape(16, 256, 64, 64)
        output = out + attn_output"""


        #combined_input = torch.cat((out, global_prompt), dim=1)

        #multihead_attn = nn.MultiheadAttention(256 , 8)
        #print(combined_input.shape)
        #output, _ = multihead_attn(combined_input, combined_input, combined_input)

        out = out + global_prompt
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)

        #global_prompt_repeated = global_prompt.repeat(1, out.shape[1], 1, 1)  # Repeat for each channel in out

        return x * (1 - mask) + (global_prompt + out) * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat




# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat