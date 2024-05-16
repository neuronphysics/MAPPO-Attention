import torch.cuda

from .slot_att_model import SlotAttentionAE
from .slot_att_trainer import SlotAttentionTrainer


def start_train_slot_att(args):
    img_size = 44
    num_slots = 4
    latent_size = 24
    encoder_params = {"width": img_size,
                      "height": img_size,
                      "channels": [32, 32, 32, 32],
                      "kernels": [5, 5, 5, 5],
                      "strides": [1, 1, 1, 1],
                      "paddings": [2, 2, 2, 2],
                      "input_channels": 3}
    decoder_params = {"width": img_size,
                      "height": img_size,
                      "input_channels": latent_size}
    model = SlotAttentionAE(name="slot_att", width=img_size, height=img_size, latent_size=latent_size,
                            encoder_params=encoder_params, decoder_params=decoder_params,
                            num_slots=num_slots)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    steps = args.slot_train_step
    optimizer_config = {}
    clip_grad_norm = 0.9
    checkpoint_steps = 10000
    logloss_steps = 1000
    logweights_steps = 1000
    logimages_steps = 1000
    logvalid_steps = 1000
    debug = True
    working_dir = args.slot_att_work_path
    model = model
    dataloaders = []
    use_exp_decay = args.slot_att_use_exp_decay
    exp_decay_rate = args.slot_att_exp_decay_rate
    exp_decay_steps = args.slot_att_exp_decay_step
    use_warmup_lr: bool = args.slot_att_use_warmup
    warmup_steps = args.slot_att_warmup_step

    trainer = SlotAttentionTrainer(device=device,
                                   steps=steps,
                                   optimizer_config=optimizer_config,
                                   clip_grad_norm=clip_grad_norm,
                                   checkpoint_steps=checkpoint_steps,
                                   logloss_steps=logloss_steps,
                                   logweights_steps=logweights_steps,
                                   logimages_steps=logimages_steps,
                                   logvalid_steps=logvalid_steps,
                                   debug=debug,
                                   working_dir=working_dir,
                                   model=model,
                                   dataloaders=dataloaders,
                                   use_exp_decay=use_exp_decay,
                                   exp_decay_rate=exp_decay_rate,
                                   exp_decay_steps=exp_decay_steps,
                                   use_warmup_lr=use_warmup_lr,
                                   warmup_steps=warmup_steps,
                                   )
