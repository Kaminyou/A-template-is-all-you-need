import os
import argparse
import math
import yaml
import json
import torch
from tensorboardX import SummaryWriter

from contrastive.data import get_dataloader
from utils import get_encoder


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    lr = cfg['lr']
    if cfg['cos_lr']:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / cfg['epochs']))
    else:  # stepwise lr schedule
        for milestone in cfg['lr_schedule']:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(args):
    # load config file
    cfg_path = os.path.join(args.experiment_directory, 'config.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # prepare model
    print('preparing model...')
    model = get_encoder(
        base_encoder_name=cfg['resnet'],
        contrastive_framework=cfg['framework'],
        pretrained=cfg['pretrained'],
        low_dim=cfg['projection_dim'],
        m=cfg['momentum'],
        T=cfg['temperature'],
        K=cfg['queue_size']
    ).cuda()
    
    # prepare optimizer
    print('preparing optimizer...')
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    
    # prepare dataloader
    print('preparing dataloader...')
    with open(cfg['split'], 'r') as f:
        split = json.load(f)
    dataloader = get_dataloader(
        cfg['datasource'], 
        split, 
        cfg['batch_size']
    )
    
    # prepare logging directories
    checkpoint_dir = os.path.join(args.experiment_directory, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True) 
    tensorboard_dir = os.path.join(args.experiment_directory, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # prepare writer
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # resume training
    start_epoch = 0
    if args.continue_from is not None:
        load_path = os.path.join(
            checkpoint_dir, 
            f'checkpoint_{args.continue_from}.pt'
        )
        checkpoint = torch.load(load_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'resume training from epoch {start_epoch}')
        
    
    # start training
    print('start training!')
    for i_epoch in range(start_epoch, cfg['epochs']):
        # train for one epoch
        loss_epoch = 0.0
        for img1, img2 in dataloader:
            # compute contrastive loss
            img1 = img1.cuda()
            img2 = img2.cuda()
            feat1, feat2, loss = model(img1, img2, return_loss=True)
            
            # compute gradient and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss
            
        # adjust learning rate
        adjust_learning_rate(optimizer, i_epoch+1, cfg)
        
        # logging
        loss_epoch /= len(dataloader)
        writer.add_scalar("loss", loss_epoch, i_epoch+1)
        print(f"Epoch [{i_epoch}/{cfg['epochs']}]\t Loss: {loss_epoch}")
        checkpoint = {
            'epoch': i_epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_path = os.path.join(checkpoint_dir, f'checkpoint_latest.pt')
        torch.save(checkpoint, save_path)
        
        # save checkpoint
        if (i_epoch+1) % cfg['save_every'] == 0:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_{i_epoch}.pt')
            torch.save(checkpoint, save_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive learning")
    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment configuration in 'config.yaml', and logging will be "
        + "done in this directory as well.",
    )
    parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    args = parser.parse_args()

    main(args)
