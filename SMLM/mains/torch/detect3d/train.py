import argparse
import collections
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

import SMLM.torch.train.loss as module_loss
import SMLM.torch.train.metrics as module_metric
import SMLM.torch.models as module_arch

from SMLM.torch.utils import ConfigParser
from SMLM.torch.utils import prepare_device
from SMLM.torch.train import LocalizationTrainer

from SMLM.generators import Mix3D
from SMLM.torch.loaders import SMLMDataLoader3D
from torchsummary import summary

train_config = 'train.json'
file = open(train_config)
train_config = json.load(file)
train_config = ConfigParser(train_config)
logger = train_config.get_logger('train')

batch_size = 4
dataloader = SMLMDataLoader3D(train_config,batch_size,validation_split=0.1,shuffle=False)
valid_data_loader = dataloader.split_validation()

model = train_config.init_obj('arch', module_arch)
logger.info(model)

n_gpu = 1
device, device_ids = prepare_device(n_gpu)
model = model.to(device)
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)

criterion = getattr(module_loss, train_config['loss'])
metrics = [getattr(module_metric, met) for met in train_config['metrics']]
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = train_config.init_obj('optimizer', torch.optim, trainable_params)
lr_scheduler = train_config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)


summary(model,(1,121,121))
trainer = LocalizationTrainer(model, criterion, metrics, optimizer,
                              config=train_config,device=device,
                              data_loader=dataloader,
                              valid_data_loader=valid_data_loader,
                              lr_scheduler=lr_scheduler)
                                          

trainer.train()

