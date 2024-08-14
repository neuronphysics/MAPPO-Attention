import argparse
import os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import optim
from onpolicy.algorithms.utils.QSA.data_loader import GlobDataset


from .model_trans_dec import SLATE
import math


def generate_model(args):
    model = SLATE(args)
    model.to(args.device)
    if args.slot_att_load_model:
        tau, sigma = load_slot_att_model(model, args)
        args.tau_start = tau
        args.sigma_start = sigma
    return model


def configure_optimizers(model, args):
    params = [
        {'params': (x[1] for x in model.named_parameters() if 'dvae' in x[0]), 'lr': args.lr_dvae},
        {'params': (x[1] for x in model.named_parameters() if 'dvae' not in x[0]), 'lr': args.lr_main},
    ]
    optimizer = optim.Adam(params)

    warmup_steps = args.warmup_steps
    decay_steps = args.decay_steps

    def lr_scheduler_dave(step: int):
        factor = 0.5 ** (step / decay_steps)
        return factor

    def lr_scheduler_main(step: int):
        if step < warmup_steps:
            factor = step / warmup_steps
        else:
            factor = 1
        factor *= 0.5 ** (step / decay_steps)
        return factor

    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=[lr_scheduler_dave, lr_scheduler_main])

    return (
        optimizer,
        scheduler,
    )


def train_qsa(args):
    train_dataset = GlobDataset(world_root=args.slot_att_work_path + "world_data/*", phase='train', img_glob="*.pt",
                                crop_repeat=args.slot_att_crop_repeat, crop_size=args.crop_size)
    # val_dataset = GlobDataset(root=args.slot_att_work_path + "world_data/*", phase='val', img_glob="*.pt",
    #                           seq_len=args.seq_len)

    loader_kwargs = {
        'batch_size': args.slot_pretrain_batch_size,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': True,
    }

    train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
    # val_loader = DataLoader(val_dataset, sampler=None, **loader_kwargs)

    if not os.path.exists(args.slot_att_work_path + args.substrate_name + "/"):
        os.makedirs(args.slot_att_work_path + args.substrate_name + "/")

    writer = SummaryWriter(args.slot_att_work_path + args.substrate_name + "/" + "tensorboard/")

    model: nn.Module = generate_model(args)
    optimizer, scheduler = configure_optimizers(model, args)

    model.train()
    for ep in tqdm(range(args.slot_train_ep)):
        for idx, batch_data in enumerate(train_loader):
            # batch, channel, height, width
            global_step = ep * len(train_loader) + idx
            tau = cosine_anneal(global_step, args.tau_steps, start_value=args.tau_start,
                                final_value=args.tau_final)
            sigma = cosine_anneal(global_step, args.sigma_steps, start_value=args.sigma_start,
                                  final_value=args.sigma_final)
            out = model(batch_data.to(args.device), tau=tau, sigma=sigma, is_Train=True,
                        visualize=ep % args.slot_log_fre == 0)

            mse_loss = out['loss']['mse']
            similarity_loss = out['sim_loss']
            cross_entropy = out['loss']['cross_entropy']

            optimizer.zero_grad()
            loss = mse_loss + cross_entropy + similarity_loss
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.slot_clip_grade_norm)

            optimizer.step()
            scheduler.step(global_step)

            writer.add_scalar("train_dvae_loss", mse_loss, global_step)
            writer.add_scalar("train_loss", cross_entropy, global_step)
            writer.add_scalar("similarity_loss", similarity_loss, global_step)

        if ep % args.slot_log_fre == 0:
            masked_image, combined_mask, recon_row = visualize_img(out, batch_data)
            writer.add_image("masked image", masked_image, global_step=ep)
            writer.add_image("recon", recon_row, global_step=ep)
            writer.add_image("atten masks", combined_mask, global_step=ep)

        if ep % args.slot_save_fre == 0:
            save_slot_att_model(model, tau, sigma, args)

def visualize_img(out, original):
    B, C, H, W = original.shape
    recon_img = out['recon'][0].cpu()
    img_mask = out['attns'][0].cpu()
    pred_img = out['pred_image'][0].cpu()
    S, _, _, _ = img_mask.shape
    slot_img = original.unsqueeze(1)[0]  # slot, channel, H, W

    masked_image = img_mask * slot_img  # slot, channel, H, W
    masked_image = masked_image.permute(1, 2, 0, 3).reshape(C, H, -1)

    recon_row = torch.cat([original[0], recon_img, pred_img], dim=-1)

    combined_mask = img_mask.permute(1, 2, 0, 3).reshape(1, H, -1)
    return masked_image, combined_mask, recon_row


def load_slot_att_model(model, args):
    if args.attention_module == "RIM":
        num_slots = args.rim_num_units
    else:
        num_slots = args.scoff_num_units
    latent_size = args.hidden_size // num_slots
    model_name = "ns_" + str(num_slots) + "_ls_" + str(latent_size) + "_model.pt"
    data_pack = torch.load(args.slot_att_work_path + args.substrate_name + "/" + model_name)
    model.load_state_dict(data_pack['model'])
    model.dvae.decoder_normalize = True
    return data_pack['tau'], data_pack['sigma']


def save_slot_att_model(model, tau, sigma, args):
    if args.attention_module == "RIM":
        num_slots = args.rim_num_units
    else:
        num_slots = args.scoff_num_units
    latent_size = args.hidden_size // num_slots
    model_name = "ns_" + str(num_slots) + "_ls_" + str(latent_size) + "_model.pt"
    data_pack = {"model": model.state_dict(), "tau": tau, "sigma": sigma}
    torch.save(data_pack, args.slot_att_work_path + args.substrate_name + "/" + model_name)


def cosine_anneal(step, final_step, start_step=0, start_value=1.0, final_value=0.1):
    assert start_value >= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = 0.5 * (start_value - final_value)
        b = 0.5 * (start_value + final_value)
        progress = (step - start_step) / (final_step - start_step)
        value = a * math.cos(math.pi * progress) + b
    return value
