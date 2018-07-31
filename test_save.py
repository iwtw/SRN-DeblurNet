import torch
import argparse
from network import SRNDeblurNet
from data import TestSaveDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model,set_requires_grad,compute_psnr
from time import time
import os
from skimage.io import imsave



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-resume')
    parser.add_argument('-input_list')
    parser.add_argument('-output_dir')
    parser.add_argument('--output_list',type=str,default='./output.list')
    parser.add_argument('--preserve_dir_layer',type=int,default=0)
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--resume_epoch',default=None)
    return parser.parse_args()


if __name__ == '__main__' :
    args = parse_args()

    img_list = open(args.input_list, 'r').read().strip().split('\n')
    dataset = TestSaveDataset(img_list)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = True )
    
    net = SRNDeblurNet().cuda()
    set_requires_grad(net,False)
    last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  ) 
    
    psnr_list = []
    output_list = []

    tt = time()
    for step , batch in tqdm(enumerate( dataloader ) , total = len(dataloader) ):
        for k in batch:
            if 'img' in k:
                batch[k] = batch[k].cuda(async = True)
                batch[k].requires_grad = False

        
        y, _ , _ = net( batch['img256'] , batch['img128'] , batch['img64'] )

        y.detach_() 
        y = ((y.clamp(-1,1) + 1.0) / 2.0 * 255.999).byte()
        y = y.permute( 0 , 2 , 3 , 1   ).cpu().numpy()#NHWC
        input_filename_list = batch['filename']
        output_filename_list = []
        for filename in input_filename_list:
            output_filename_list.append( '/'.join( [args.output_dir] + filename.split('/')[ -1 - args.preserve_dir_layer : ] ) )
        output_list += output_filename_list
        assert len(output_filename_list) == len(y)
        filename_iter = iter(output_filename_list)
        for img in y:
            filename = next(filename_iter)
            if not os.path.exists('/'.join( filename.split('/')[:-1] ) ) :
                os.system('mkdir -p {}'.format( '/'.join( filename.split('/')[:-1] )) )
            imsave( filename , img )
    
    with open(args.output_list,'w') as fp:
        fp.write( '\n'.join( output_list ) + '\n' )
            

