# import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from .helper_function import *

# import timm

import logging

from scipy import ndimage

from model.decoders import HADer
from model.maxxvit_4out import maxvit_tiny_rw_224 as maxvit_tiny_rw_224_4out
from model.maxxvit_4out import maxvit_rmlp_tiny_rw_256 as maxvit_rmlp_tiny_rw_256_4out
from model.maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from model.maxxvit_4out import maxvit_rmlp_small_rw_224 as maxvit_rmlp_small_rw_224_4out
from timm.models.layers import trunc_normal_
import math



logger = logging.getLogger(__name__)
def load_pretrained_weights(img_size, model_scale, pre_load):
    if(model_scale=='tiny'):
        if img_size==224:
            backbone = maxvit_tiny_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './model/maxvit_tiny_rw_224_sw-7d0dffeb.pth')
            state_dict = torch.load('path-to-pretrained_model.pth')
        elif(img_size==256):
            backbone = maxvit_rmlp_tiny_rw_256_4out()
            print('Loading:', './model/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
            state_dict = torch.load('./model/maxvit_rmlp_tiny_rw_256_sw-bbef0ff5.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")

    elif(model_scale=='small'):
        if img_size==224:
            backbone = maxvit_rmlp_small_rw_224_4out()  # [64, 128, 320, 512]
            print('Loading:', './model/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
            state_dict = torch.load('./model/maxvit_rmlp_small_rw_224_sw-6ef0ae4f.pth')
        elif(img_size==256):
            backbone = maxxvit_rmlp_small_rw_256_4out()
            print('Loading:', './model/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
            state_dict = torch.load('./model/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth')
        else:
            sys.exit(str(img_size)+" is not a valid image size! Currently supported image sizes are 224 and 256.")
    else:
        sys.exit(model_scale+" is not a valid model scale! Currently supported model scales are 'tiny' and 'small'.")
    if pre_load == True:    
        backbone.load_state_dict(state_dict, strict=False)
        print('Pretrain weights loaded.')
    
    return backbone





class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights



class SpatialWeights(nn.Module):
    def __init__(self, dim=48,ffn_expansion_factor=0.25, bias=False):
        super(SpatialWeights, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(2*dim, hidden_features*3, kernel_size=(1,1,1), bias=bias)

        self.dwconv1 = nn.Conv3d(hidden_features, hidden_features, kernel_size=(3,3,3), stride=1, dilation=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=2, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=(3,3), stride=1, dilation=3, padding=3, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, 2*dim, kernel_size=(1,1,1), bias=bias)



    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        x = x.unsqueeze(2)
        x = self.project_in(x)
        x1,x2,x3 = x.chunk(3, dim=1)
        x1 = self.dwconv1(x1).squeeze(2)
        x2 = self.dwconv2(x2.squeeze(2))
        x3 = self.dwconv3(x3.squeeze(2))
        x = F.gelu(x1)*(x2+x3)
        x = x.unsqueeze(2)
        x = self.project_out(x)
        x = x.squeeze(2).chunk(2, dim=1)
        return x

class CFIM(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(CFIM, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim,ffn_expansion_factor=0.5, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2  + self.lambda_s * spatial_weights[1] 
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1  + self.lambda_s * spatial_weights[0] 
        return out_x1, out_x2


class NormalGamma(nn.Module):

    def __init__(self, embed_dim,out_dim=4):
        super().__init__()

        self.net = nn.Conv2d(embed_dim, out_dim, 1)

    def forward(self, x):
        # print(x.size())
        out = self.net(x)

        mu, logv, logalpha, logbeta = torch.chunk(out, 4, dim=1)

        mu = mu

        v = F.softplus(logv) + 1e-6
        alpha = F.softplus(logalpha) + 1.0
        beta = F.softplus(logbeta) + 0.5

        return mu, v, alpha, beta

class SPD(nn.Module):
    def __init__(self, img_size_s1=(224, 224), img_size_s2=(224, 224), model_scale='tiny', pre_load=True):
        super(SPD, self).__init__()

        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale


        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.backbone1 = load_pretrained_weights(self.img_size_s1[0], self.model_scale, pre_load=pre_load)
        self.backbone2 = load_pretrained_weights(self.img_size_s2[0], self.model_scale, pre_load=pre_load)

        if (self.model_scale == 'tiny'):
            self.channels = [512, 256, 128, 64]
        elif (self.model_scale == 'small'):
            self.channels = [768, 384, 192, 96]

        self.decoder = HADer(channels=self.channels)

        self.out_head1 = nn.Conv2d(self.channels[0], 1, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], 1, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], 1, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], 1, 1)

        self.out_head1_v = nn.Conv2d(self.channels[0], 1, 1)
        self.out_head2_v = nn.Conv2d(self.channels[1], 1, 1)
        self.out_head3_v = nn.Conv2d(self.channels[2], 1, 1)
        self.out_head4_v = nn.Conv2d(self.channels[3], 1, 1)

        self.out_head1_alpha = nn.Conv2d(self.channels[0], 1, 1)
        self.out_head2_alpha = nn.Conv2d(self.channels[1], 1, 1)
        self.out_head3_alpha = nn.Conv2d(self.channels[2], 1, 1)
        self.out_head4_alpha = nn.Conv2d(self.channels[3], 1, 1)

        self.out_head1_beta = nn.Conv2d(self.channels[0], 1, 1)
        self.out_head2_beta = nn.Conv2d(self.channels[1], 1, 1)
        self.out_head3_beta = nn.Conv2d(self.channels[2], 1, 1)
        self.out_head4_beta = nn.Conv2d(self.channels[3], 1, 1)






        self.int_head11 = NormalGamma(self.channels[0],4*self.channels[0])
        self.int_head12 = NormalGamma(self.channels[1],4*self.channels[1])
        self.int_head13 = NormalGamma(self.channels[2],4*self.channels[2])
        self.int_head14 = NormalGamma(self.channels[3],4*self.channels[3])

        self.int_head21 = NormalGamma(self.channels[0], 4 * self.channels[0])
        self.int_head22 = NormalGamma(self.channels[1], 4 * self.channels[1])
        self.int_head23 = NormalGamma(self.channels[2], 4 * self.channels[2])
        self.int_head24 = NormalGamma(self.channels[3], 4 * self.channels[3])



        self.CFIMs = nn.ModuleList([
            CFIM(dim=self.channels[3], reduction=1),
            CFIM(dim=self.channels[2], reduction=1),
            CFIM(dim=self.channels[1], reduction=1),
            CFIM(dim=self.channels[0], reduction=1)])

    def NIG_fusion_two(self,
                       mu1, v1, alpha1, beta1,
                       mu2, v2, alpha2, beta2):

        v_fused = v1 + v2

        mu_fused = (v1 * mu1 + v2 * mu2) / (v_fused + 1e-8)

        alpha_fused = alpha1 + alpha2 - 1.0
        alpha_fused = torch.clamp(alpha_fused, min=1.0001)

        conflict_term = 0.5 * (v1 * v2 / (v_fused + 1e-8)) * (mu1 - mu2) ** 2

        beta_fused = beta1 + beta2 + conflict_term
        beta_fused = torch.clamp(beta_fused, min=1e-6)

        return mu_fused, v_fused, alpha_fused, beta_fused

    def NIG_fusion_four(self,
                        mu1, v1, alpha1, beta1,
                        mu2, v2, alpha2, beta2,
                        mu3, v3, alpha3, beta3,
                        mu4, v4, alpha4, beta4):

        v_fused = v1 + v2 + v3 + v4 + 1e-8

        mu_fused = (v1 * mu1 + v2 * mu2 + v3 * mu3 + v4 * mu4) / v_fused

        alpha_fused = alpha1 + alpha2 + alpha3 + alpha4 - 3.0
        alpha_fused = torch.clamp(alpha_fused, min=1.0001)


        conflict_term = 0.5 * (
                v1 * (mu1 - mu_fused) ** 2 +
                v2 * (mu2 - mu_fused) ** 2 +
                v3 * (mu3 - mu_fused) ** 2 +
                v4 * (mu4 - mu_fused) ** 2
        )

        beta_fused = beta1 + beta2 + beta3 + beta4 + conflict_term
        beta_fused = torch.clamp(beta_fused, min=1e-6)

        return mu_fused, v_fused, alpha_fused, beta_fused

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = self.conv_2(x2)

        if (x1.shape[2] % 14 != 0):
            f1 = self.backbone1(F.interpolate(x1, size=self.img_size_s1, mode='bilinear'))
        else:
            f1 = self.backbone2(F.interpolate(x1, size=self.img_size_s1, mode='bilinear'))

        if (x1.shape[2] % 14 != 0):
            f2 = self.backbone2(F.interpolate(x2, size=self.img_size_s2, mode='bilinear'))
        else:
            f2 = self.backbone1(F.interpolate(x2, size=self.img_size_s2, mode='bilinear'))

        f1_0, f2_0 = self.CFIMs[0](f1[0], f2[0])
        f1_1, f2_1 = self.CFIMs[1](f1[1], f2[1])
        f1_2, f2_2 = self.CFIMs[2](f1[2], f2[2])
        f1_3, f2_3 = self.CFIMs[3](f1[3], f2[3])

        x11_o, x12_o, x13_o, x14_o = self.decoder(f1_3, [f1_2, f1_1, f1_0])

        self.x11_mu, self.x11_v, self.x11_alpha, self.x11_beta = self.int_head11(x11_o)
        self.x12_mu, self.x12_v, self.x12_alpha, self.x12_beta = self.int_head12(x12_o)
        self.x13_mu, self.x13_v, self.x13_alpha, self.x13_beta = self.int_head13(x13_o)
        self.x14_mu, self.x14_v, self.x14_alpha, self.x14_beta = self.int_head14(x14_o)


        p11 = self.out_head1(self.x11_mu)
        p12 = self.out_head2(self.x12_mu)
        p13 = self.out_head3(self.x13_mu)
        p14 = self.out_head4(self.x14_mu)


        p11 = F.interpolate(p11, scale_factor=32, mode='bilinear')
        p12 = F.interpolate(p12, scale_factor=16, mode='bilinear')
        p13 = F.interpolate(p13, scale_factor=8, mode='bilinear')
        p14 = F.interpolate(p14, scale_factor=4, mode='bilinear')


        x21_o, x22_o, x23_o, x24_o = self.decoder(f2_3, [f2_2, f2_1, f2_0])

        self.x21_mu, self.x21_v, self.x21_alpha, self.x21_beta = self.int_head21(x21_o)
        self.x22_mu, self.x22_v, self.x22_alpha, self.x22_beta = self.int_head22(x22_o)
        self.x23_mu, self.x23_v, self.x23_alpha, self.x23_beta = self.int_head23(x23_o)
        self.x24_mu, self.x24_v, self.x24_alpha, self.x24_beta = self.int_head24(x24_o)

        p21 = self.out_head1(self.x21_mu)
        p22 = self.out_head2(self.x22_mu)
        p23 = self.out_head3(self.x23_mu)
        p24 = self.out_head4(self.x24_mu)



        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode='bilinear')
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode='bilinear')
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode='bilinear')
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode='bilinear')




        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24




        return p1 + p2 + p3 + p4

    def KL_NIG(self,mu1, v1, a1, b1, mu2, v2, a2, b2):

        eps = 1e-8

        v1 = torch.clamp(v1, min=eps)
        v2 = torch.clamp(v2, min=eps)
        b1 = torch.clamp(b1, min=eps)
        b2 = torch.clamp(b2, min=eps)
        a1 = torch.clamp(a1, min=1.0 + eps)
        a2 = torch.clamp(a2, min=1.0 + eps)

        KL = (
                0.5 * (a1 - 1) / b1 * (v2 * (mu2 - mu1) ** 2)
                + 0.5 * v2 / v1
                - 0.5 * torch.log(v2 / v1)
                - 0.5
                + a2 * torch.log(b1 / b2)
                - (torch.lgamma(a1) - torch.lgamma(a2))
                + (a1 - a2) * torch.digamma(a1)
                - (b1 - b2) * a1 / b1
        )

        return KL
    def compute_loss(self,mu,v,alpha,beta):
        prior_mu = torch.zeros_like(mu)
        prior_v = torch.ones_like(v) * 0.1
        prior_alpha = torch.ones_like(alpha) * 1.1
        prior_beta = torch.ones_like(beta) * 1.0

        KL = self.KL_NIG(mu,v,alpha,beta,prior_mu,prior_v,prior_alpha,prior_beta)
        return KL

    def get_loss(self):
        nig_loss_1 = self.compute_loss(self.x11_mu,self.x11_v,self.x11_alpha,self.x11_beta) + \
        self.compute_loss(self.x21_mu,self.x21_v,self.x21_alpha,self.x21_beta)

        nig_loss_2 = self.compute_loss(self.x12_mu, self.x12_v, self.x12_alpha, self.x12_beta) + \
                     self.compute_loss(self.x22_mu, self.x22_v, self.x22_alpha, self.x22_beta)

        nig_loss_3 = self.compute_loss(self.x13_mu, self.x13_v, self.x13_alpha, self.x13_beta) + \
                     self.compute_loss(self.x23_mu, self.x23_v, self.x23_alpha, self.x23_beta)

        nig_loss_4 = self.compute_loss(self.x14_mu, self.x14_v, self.x14_alpha, self.x14_beta) + \
            self.compute_loss(self.x24_mu,self.x24_v,self.x24_alpha,self.x24_beta)

        return (nig_loss_1.mean() + nig_loss_2.mean() +nig_loss_3.mean() +nig_loss_4.mean())

    def compute_coeff(self, v, alpha, beta, eps=1e-6):

        ua = beta / (alpha - 1 + eps)
        ue = beta / (v * (alpha - 1) + eps)
        total_u = ua + ue  # [B,C,H,W]

        total_u = total_u.mean(dim=(1, 2, 3))  # [B]

        # log scale
        log_u = torch.log(total_u + 1.0)
        coeff = torch.exp(-0.5 * log_u)

        coeff = 5.0 * coeff.detach()

        coeff = torch.clamp(coeff, max=2.0)

        return coeff
    def get_coeff(self):
        coeff_h11 = self.compute_coeff(self.x11_v,self.x11_alpha,self.x11_beta)
        coeff_h12 = self.compute_coeff(self.x12_v,self.x12_alpha,self.x12_beta)
        coeff_h13 = self.compute_coeff(self.x13_v,self.x13_alpha,self.x13_beta)
        coeff_h14 = self.compute_coeff(self.x14_v,self.x14_alpha,self.x14_beta)

        coeff_h21 = self.compute_coeff(self.x21_v,self.x21_alpha,self.x21_beta)
        coeff_h22 = self.compute_coeff(self.x22_v,self.x22_alpha,self.x22_beta)
        coeff_h23 = self.compute_coeff(self.x23_v,self.x23_alpha,self.x23_beta)
        coeff_h24 = self.compute_coeff(self.x24_v,self.x24_alpha,self.x24_beta)

        return {
            'int_head11':coeff_h11,
            'int_head12':coeff_h12,
            'int_head13':coeff_h13,
            'int_head14':coeff_h14,
            'int_head21':coeff_h21,
            'int_head22':coeff_h22,
            'int_head23':coeff_h23,
            'int_head24':coeff_h24,

        }


