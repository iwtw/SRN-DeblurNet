import torch
from utils import *
import network
import train_config as config


stem = network.resnet18( halving = config.stem['halving'] , num_classes = config.stem['num_classes'] , feature_layer_dim = config.stem['feature_layer_dim'] , use_batchnorm  = config.stem['use_batchnorm'] , dream = None , preactivation = config.stem['preactivation'] , use_avgpool = config.stem['use_avgpool'])

optimizer = torch.optim.SGD( stem.parameters() , lr = 1.0 , weight_decay = config.loss['weight_l2_reg'] , momentum = config.train['momentum'] , nesterov = config.train['nesterov'] )

resume_optimizer( optimizer , 'save/stitching_train/try_17')
print( optimizer.param_groups[0]['lr'] )
        
