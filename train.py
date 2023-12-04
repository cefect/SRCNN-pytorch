
import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

from definitions import wrk_dir



def train_model(train_file,
                eval_file,
                out_dir=None, 
                
                scale=3,
                lr=1e-4,
                batch_size=16,
                num_epochs=400,
                num_workers=8,
                seed=123,
                
 
 
                ):
    
    """train the SCRNN aginst a dataset with labels
    
    Params
    --------
    train_file: str
    
    eval_file: str
    
    out_dir: str
    
    scale: int
        upscaling value
        
    lr: float
        optimizer's learning rate
        
    batch_size: int
        size to batch data into
        
    num_epochs: int
        number of training epochs
        
    num_workers: int
        number of works for loading data in parallel
        
        
    seed: int
        random seed for initilizeing paramaters
        
    """
    #===========================================================================
    # setup
    #===========================================================================
    #configure outputs
    if out_dir is None:
        out_dir = os.path.join(wrk_dir, 'outs', 'SRCNN', f'x{scale}')
    
 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        #set device
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'running on device: {device}')
    torch.manual_seed(seed)
    #===========================================================================
    # initialize
    #===========================================================================
    print('init model')
    model = SRCNN(num_channels=1).to( #only training on y-channel
        device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
            {'params':model.conv1.parameters()}, 
            {'params':model.conv2.parameters()}, 
            {'params':model.conv3.parameters(), 'lr':lr * 0.1}], 
        lr=lr)
    #===========================================================================
    # data
    #===========================================================================
    print(f'init data')
    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True)

    #eval data
    eval_dataset = EvalDataset(eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    
    #===========================================================================
    # train loop
    #===========================================================================
    print(f'start training {num_epochs} loops\n-------------------')
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    backprops = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))
            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                backprops += 1
        
        #=======================================================================
        # write these epoch parameters
        #=======================================================================
        model_fp = os.path.join(out_dir, 'epoch_{}.pth'.format(epoch))
        print(f'saving model state to {model_fp}')
        torch.save(model.state_dict(), model_fp)
        #=======================================================================
        # evaluate this epoch
        #=======================================================================
        print(f'evaluating against {batch_size} batches')
        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
        
        print('eval psnr: {:.2f} (backprops={})'.format(epoch_psnr.avg, backprops))
        #wrap eval
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
    
    #===========================================================================
    # wrap
    #===========================================================================
    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    ofp = os.path.join(out_dir, 'best.pth')
    torch.save(best_weights, ofp)
    print(f'saved to \n    ofp')
    
    return ofp






if __name__ == '__main__':
    #set the argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=False, default=None)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='data batches')
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoading subprocesses to use for data            loading')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    
    train_model(args.train_file, args.eval_file, out_dir=args.out_dir, scale=args.scale,
                lr=args.lr, batch_size=args.batch_size, num_epochs=args.num_epochs, num_workers=args.num_workers,
                seed=args.seed)
    
    
    
    
    
    
    
    
    
    
    
