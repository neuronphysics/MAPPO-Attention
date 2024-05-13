import math
import os.path
import argparse

import torch
import torchvision.utils as vutils
import wandb
from torch import arange as ar
from torch.optim import Adam
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from datetime import datetime

from .steve import STEVE
from .data import GlobVideoDataset
from .utils import cosine_anneal, linear_warmup


def visualize(video, recon_dvae, att_mask, N=1, num_slots=15):
    B, T, C, H, W = video.size()

    frames = []
    for t in range(T):
        video_t = video[:N, t, None, :, :, :]
        recon_dvae_t = recon_dvae[:N, t, None, :, :, :]
        att_mask_t = att_mask[:N, t, :, :, :, :].repeat(1, 1, 3, 1, 1)

        att_mask_t[att_mask_t <= 0.5] = 0
        att_mask_t[att_mask_t > 0.5] = 1
        masked_video_t = video_t * att_mask_t

        total_mask = att_mask_t.sum(dim=1)
        total_mask = total_mask.unsqueeze(1)
        total_mask[total_mask > 1] = 1
        total_mask_video = video_t * total_mask

        # tile
        tiles = torch.cat((video_t, recon_dvae_t, total_mask_video, masked_video_t), dim=1).flatten(end_dim=1)

        # grid
        frame = vutils.make_grid(tiles, nrow=(num_slots + 3), pad_value=0.8)
        frames += [frame]

    frames = torch.stack(frames, dim=0).unsqueeze(0)

    return frames


def train(args):
    def log_scaler(name, value, step):
        if use_wandb:
            wandb.log({name: value}, step)
        else:
            writer.add_scalar(name, value, step)
    
    def log_video(name, frames):
        if use_wandb:
            wandb.log({"video": wandb.Video(frames.cpu().numpy()[0], fps=4, format="gif")})
        else:
            writer.add_video(name, frames)
    
    torch.manual_seed(args.seed)
    num_slots = args.num_slots
    use_wandb = args.use_wandb

    arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
    arg_str = '__'.join(arg_str_list)
    log_dir = os.path.join(args.log_path, datetime.today().isoformat())
    writer = SummaryWriter(log_dir)
    writer.add_text('hparams', arg_str)

    train_dataset = GlobVideoDataset(root=args.data_path, phase='train', ep_len=args.steve_video_seq_len, )
    val_dataset = GlobVideoDataset(root=args.data_path, phase='val', ep_len=args.steve_video_seq_len, )

    loader_kwargs = {
        'batch_size': args.steve_pretrain_batch_size,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'drop_last': True,
    }

    train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=None, **loader_kwargs)

    train_epoch_size = len(train_loader)
    val_epoch_size = len(val_loader)

    log_interval = max(train_epoch_size // 5, 1)

    train_sample = next(iter(train_loader))
    img_size = train_sample.shape[-1]
    args.image_size = img_size

    model = STEVE(args)

    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['best_epoch']
        model.load_state_dict(checkpoint['model'])
    else:
        checkpoint = None
        start_epoch = 0
        best_val_loss = math.inf
        best_epoch = 0

    model = model.cuda()
    if args.use_dp:
        model = DP(model)

    optimizer = Adam([
        {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
        {'params': (x[1] for x in model.named_parameters() if 'steve_encoder' in x[0]), 'lr': 0.0},
        {'params': (x[1] for x in model.named_parameters() if 'steve_decoder' in x[0]), 'lr': 0.0},
    ])

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, args.epochs):
        model.train()

        for batch, video in enumerate(train_loader):
            global_step = epoch * train_epoch_size + batch

            tau = cosine_anneal(
                global_step,
                args.tau_start,
                args.tau_final,
                0,
                args.tau_steps)

            lr_warmup_factor_enc = linear_warmup(
                global_step,
                0.,
                1.0,
                0.,
                args.lr_warmup_steps)

            lr_warmup_factor_dec = linear_warmup(
                global_step,
                0.,
                1.0,
                0,
                args.lr_warmup_steps)

            lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))

            optimizer.param_groups[0]['lr'] = args.lr_dvae
            optimizer.param_groups[1]['lr'] = lr_decay_factor * lr_warmup_factor_enc * args.lr_enc
            optimizer.param_groups[2]['lr'] = lr_decay_factor * lr_warmup_factor_dec * args.lr_dec

            video = video.cuda()

            optimizer.zero_grad()

            (recon, cross_entropy, mse, att_mask) = model(video, tau, args.hard)

            if args.use_dp:
                mse = mse.mean()
                cross_entropy = cross_entropy.mean()

            loss = mse + cross_entropy

            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip, 'inf')
            optimizer.step()

            with torch.no_grad():
                if batch % log_interval == 0:
                    print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F}'.format(
                        epoch + 1, batch, train_epoch_size, loss.item(), mse.item()))

                    log_scaler('TRAIN/loss', loss.item(), global_step)
                    log_scaler('TRAIN/cross_entropy', cross_entropy.item(), global_step)
                    log_scaler('TRAIN/mse', mse.item(), global_step)

                    log_scaler('TRAIN/tau', tau, global_step)
                    log_scaler('TRAIN/lr_dvae', optimizer.param_groups[0]['lr'], global_step)
                    log_scaler('TRAIN/lr_enc', optimizer.param_groups[1]['lr'], global_step)
                    log_scaler('TRAIN/lr_dec', optimizer.param_groups[2]['lr'], global_step)

        if epoch % 10 == 0:
            with torch.no_grad():
                frames = visualize(video, recon, att_mask, N=1, num_slots=num_slots)
                log_video('TRAIN_recons/epoch={:03}'.format(epoch + 1), frames)

        with torch.no_grad():
            model.eval()

            val_cross_entropy = 0.
            val_mse = 0.

            for batch, video in enumerate(val_loader):
                video = video.cuda()

                (recon, cross_entropy, mse, att_mask) = model(video, tau, args.hard)

                if args.use_dp:
                    mse = mse.mean()
                    cross_entropy = cross_entropy.mean()

                val_cross_entropy += cross_entropy.item()
                val_mse += mse.item()

            val_cross_entropy /= (val_epoch_size)
            val_mse /= (val_epoch_size)

            val_loss = val_mse + val_cross_entropy

            log_scaler('VAL/loss', val_loss, epoch + 1)
            log_scaler('VAL/cross_entropy', val_cross_entropy, epoch + 1)
            log_scaler('VAL/mse', val_mse, epoch + 1)

            print('====> Epoch: {:3} \t Loss = {:F}'.format(epoch + 1, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                torch.save(model.module.state_dict() if args.use_dp else model.state_dict(),
                           os.path.join(log_dir, 'best_model.pt'))

                if global_step < args.steps:
                    torch.save(model.module.state_dict() if args.use_dp else model.state_dict(),
                               os.path.join(log_dir, f'best_model_until_{args.steps}_steps.pt'))

                if 50 <= epoch:
                    frames = visualize(video, recon, att_mask, N=1, num_slots=num_slots)
                    log_video('VAL_recons/epoch={:03}'.format(epoch + 1), frames)

            log_scaler('VAL/best_loss', best_val_loss, epoch + 1)

            checkpoint = {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'model': model.module.state_dict() if args.use_dp else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))

            print('====> Best Loss = {:F} @ Epoch {}'.format(best_val_loss, best_epoch))

    writer.close()


