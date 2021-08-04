import torch
from torch import nn
from torch.nn import functional as F
from vdvae.vae_helpers import HModule, get_1x1, get_3x3, DmolNet, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools

from vdvae.train import *
from vdvae.hps import *

class SRVAE(nn.Module):
  def build(self):
    a=1	##################################################################################

  def build_model(self,image_size, net_type = None):
    self.H1, self.logprint1 = set_up_hyperparams()
    self.H1.image_size = image_size
    if net_type is not None:
      self.H1.update(net_type)
    self.H1.image_channels = 3
    self.vae, self.ema_vae = load_vaes(self.H1, self.logprint1)

  def build_parcial_model(self, image_size, net_type = None):
    self.H2, self.logprint2 = set_up_hyperparams()
    self.H2.image_size = image_size
    self.H2.image_channels = 3
    n_batch = self.H2.n_batch
    if net_type is not None:
      self.H2.update(net_type)
    self.H2.n_batch = n_batch
    self.vae_sr, self.ema_vae_sr = load_vaes(self.H2, self.logprint2)

  def load_saved_models(self, model_path, model_path_ema):
		
    model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location="cuda").items()}
    model_state_dict = self.vae.state_dict()
    model_state_dict.update(model_state_dict_save)
    self.vae.load_state_dict(model_state_dict)

    model_state_dict_save = {k:v for k,v in torch.load(model_path_ema, map_location="cuda").items()}
    model_state_dict = self.ema_vae.state_dict()
    model_state_dict.update(model_state_dict_save)
    self.ema_vae.load_state_dict(model_state_dict)

  def load_saved_models_sr(self, model_path_sr, model_path_ema_sr):

    model_state_dict_save = {k:v for k,v in torch.load(model_path_sr, map_location="cuda").items()}
    model_state_dict = self.vae_sr.state_dict()
    model_state_dict.update(model_state_dict_save)
    self.vae_sr.load_state_dict(model_state_dict)

    model_state_dict_save = {k:v for k,v in torch.load(model_path_ema_sr, map_location="cuda").items()}
    model_state_dict = self.ema_vae_sr.state_dict()
    model_state_dict.update(model_state_dict_save)
    self.ema_vae_sr.load_state_dict(model_state_dict)


  def forward(self, lr, hr):
    lr, lr_proc = self.preprocess_func(lr)
    #lr = lr.permute(0, 2, 3, 1).contiguous()
    activations_sr = self.vae_sr.module.forward_sr_activations(lr)
    hr, hr_proc = self.preprocess_func(hr)
    stats = self.vae.forward(hr, hr_proc, activations_sr=activations_sr)
    return stats
        
  def forward_ema(self,lr,hr):
    lr, lr_proc = self.preprocess_func(lr)
    #lr = lr.permute(0, 2, 3, 1).contiguous()
    activations_sr = self.ema_vae_sr.forward_sr_activations(lr)
    hr, hr_proc = self.preprocess_func(hr)
    stats = self.ema_vae.forward(hr, hr_proc, activations_sr=activations_sr)
    return stats

  def forward_sr_sample(self, x, n_batch):
    x, x_proc = self.preprocess_func(x)
    activations_sr = self.ema_vae_sr.forward_sr_activations(x)
    
    output = self.ema_vae.forward_sr_sample(n_batch, activations_sr)
    return output

  def preprocess_func(self,x):
    shift = -115.92961967
    scale = 1. / 69.37404
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)
    'takes in a data example and returns the preprocessed input'
    'as well as the input processed for the loss'
    x = x * 255
    x = x.type(torch.ByteTensor)
    x = x.permute(0, 2, 3, 1)

    inp = x.cuda(non_blocking=True).float()
    out = inp.clone()
    inp.add_(shift).mul_(scale)

    #Does low-bit here
    #out.mul_(1. / 8.).floor_().mul_(8.)

    out.add_(shift_loss).mul_(scale_loss)
    return inp, out


class conv_net_partial(nn.Module):
  def build(self):
    self.cnn_1 = nn.Conv2d(in_channels=3,
                                out_channels=512,
                                kernel_size=1,
                                stride=1,
                                padding=0)

    self.cnn_2 = nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=9,
                                stride=1,
                                padding=0)

    self.cnn_3 = nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=5,
                                stride=1,
                                padding=0)

    self.cnn_4 = nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=4,
                                stride=1,
                                padding=0)
  def forward(self,x):
    x1 = F.relu(self.cnn_1(x))
    x2 = F.relu(self.cnn_2(x1))
    x3 = F.relu(self.cnn_3(x2))
    x4 = F.relu(self.cnn_4(x3))
    return {1:x4,4:x3,8:x2,16:x1}

class SRVAE_Small(nn.Module):
  def build(self):
    a=1 ##################################################################################

  def build_model(self,image_size, net_type = None):
    self.H1, self.logprint1 = set_up_hyperparams()
    self.H1.image_size = image_size
    if net_type is not None:
      self.H1.update(net_type)
    self.H1.image_channels = 3
    self.vae, self.ema_vae = load_vaes(self.H1, self.logprint1)

  def build_parcial_model(self):
    #self.vae_sr = nn.Sequential(
    #        nn.ReLU(),
    #        nn.Conv2d(in_channels=3,
    #                           out_channels=512,
    #                           kernel_size=16,
    #                           stride=1,
    #                           padding=0)
    #    )
    self.vae_sr = conv_net_partial()
    self.vae_sr.build()
    self.vae_sr = self.vae_sr.cuda()

  def load_saved_models(self, model_path, model_path_ema):
    
    model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location="cuda").items()}
    model_state_dict = self.vae.state_dict()
    model_state_dict.update(model_state_dict_save)
    self.vae.load_state_dict(model_state_dict)

    model_state_dict_save = {k:v for k,v in torch.load(model_path_ema, map_location="cuda").items()}
    model_state_dict = self.ema_vae.state_dict()
    model_state_dict.update(model_state_dict_save)
    self.ema_vae.load_state_dict(model_state_dict)

  def forward(self, lr, hr):
    #lr, lr_proc = self.preprocess_func(lr)
    #lr = lr.permute(0, 2, 3, 1).contiguous()
    activations_sr = self.vae_sr(lr)
    hr, hr_proc = self.preprocess_func(hr)
    stats = self.vae.forward(hr, hr_proc, activations_sr=activations_sr)
    return stats
        
  def forward_ema(self,lr,hr):
    #lr, lr_proc = self.preprocess_func(lr)
    #lr = lr.permute(0, 2, 3, 1).contiguous()
    activations_sr = self.vae_sr(lr)
    hr, hr_proc = self.preprocess_func(hr)
    stats = self.ema_vae.forward(hr, hr_proc, activations_sr=activations_sr)
    return stats

  def forward_sr_sample(self, x, n_batch):
    #x, x_proc = self.preprocess_func(x)
    activations_sr = self.vae_sr(x)
    
    output = self.ema_vae.forward_sr_sample(n_batch, activations_sr)
    return output

  def preprocess_func(self,x):
    shift = -115.92961967
    scale = 1. / 69.37404
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)
    'takes in a data example and returns the preprocessed input'
    'as well as the input processed for the loss'
    x = x * 255
    x = x.type(torch.ByteTensor)
    x = x.permute(0, 2, 3, 1)

    inp = x.cuda(non_blocking=True).float()
    out = inp.clone()
    inp.add_(shift).mul_(scale)

    #Does low-bit here
    #out.mul_(1. / 8.).floor_().mul_(8.)

    out.add_(shift_loss).mul_(scale_loss)
    return inp, out