import torch
import argparse
from network import SRNDeblurNet
from data import TestDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model,set_requires_grad,compute_psnr
from time import time
import os



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('resume')
    parser.add_argument('--gamma',dest='gamma',action='store_true')
    parser.set_defaults(gamma=False)
    parser.add_argument('--batch_size' , type=int,default=1)
    parser.add_argument('--resume_epoch',default=None)
    return parser.parse_args()


if __name__ == '__main__' :
    args = parse_args()
    assert args.dataset in ['gopro']
    if args.dataset == 'gopro':
        if args.gamma:
            input_list = 'test_gopro_gamma.list'
        else:
            input_list = 'test_gopro.list'


    img_list = open(input_list, 'r').read().strip().split('\n')
    dataset = TestDataset(img_list)
    dataloader = DataLoader( dataset , batch_size = args.batch_size , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = True )
    
    net = SRNDeblurNet().cuda()
    set_requires_grad(net,False)
    last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  ) 
    
    log_dir = '{}/test/{}'.format(args.resume,args.dataset)
    os.system('mkdir -p {}'.format(log_dir) )
    psnr_list = []

    tt = time()
    with torch.no_grad():
        for step , batch in tqdm(enumerate( dataloader ) , total = len(dataloader) ):
            for k in batch:
                batch[k] = batch[k].cuda(async = True)
                batch[k].requires_grad = False

            y256 , y128 , y64 = net( batch['img256'] , batch['img128'] , batch['img64'] )
            if step==0:
                print(y256.shape)
            psnr_list.append( compute_psnr(y256 ,  batch['label256'], 2 ).cpu().numpy() )
            if step % 100 == 100 -1 :
                t = time()
                psnr = np.mean( psnr_list )
                tqdm.write("{} / {} : psnr {} , {} img/s".format( step , len(dataloader) - 1 , psnr , 100*args.batch_size / (t-tt)   ) )
                tt = t
    psnr = np.mean( psnr_list )
    print( psnr )

    with open('{}/psnr.txt'.format(log_dir),'a') as log_fp:
        log_fp.write( 'epoch {} : psnr {}'.format( last_epoch , psnr ) )
