import torch.cuda

from .slot_att_model import SlotAttentionAE
from .slot_att_trainer import SlotAttentionTrainer
from .data_loader import GlobDataset
from torch.utils.data import DataLoader


def generate_model(args):
    img_size = 44

    if args.attention_module == "RIM":
        num_slots = args.rim_num_units
    else:
        num_slots = args.scoff_num_units

    latent_size = args.hidden_size // num_slots
    encoder_params = {"width": img_size,
                      "height": img_size,
                      "channels": [32, 32, 32, 32],
                      "kernels": [5, 5, 5, 5],
                      "strides": [1, 1, 1, 1],
                      "paddings": [2, 2, 2, 2],
                      "input_channels": 3}

    decoder_params = {"width": img_size,
                      "height": img_size,
                      "input_channels": latent_size,
                      "conv_transposes": False,
                      "channels": [32, 32, 32, 4],
                      "kernels": [5, 5, 5, 3],
                      "strides": [1, 1, 1, 1],
                      "paddings": [2, 2, 2, 1],
                      "output_paddings": [0, 0, 0, 0],
                      "activations": ["relu", "relu", "relu", None],
                      }
    model = SlotAttentionAE(name="slot-attention", width=img_size, height=img_size, latent_size=latent_size,
                            encoder_params=encoder_params, decoder_params=decoder_params,
                            num_slots=num_slots)

    if args.slot_att_load_model:
        load_slot_att_model(model, args)

    return model


def load_slot_att_model(model, args):
    if args.attention_module == "RIM":
        num_slots = args.rim_num_units
    else:
        num_slots = args.scoff_num_units
    latent_size = args.hidden_size // num_slots
    model_name = "ns_" + num_slots + "_ls_" + latent_size + "_model.pt"
    model_state_dict = torch.load(args.slot_att_work_path + model_name)
    model.load_state_dict(model_state_dict)


def start_train_slot_att(args):
    model = generate_model(args)

    train_dataset = GlobDataset(root=args.slot_att_work_path + "data/*", phase='train', img_glob="*.pt")
    val_dataset = GlobDataset(root=args.slot_att_work_path + "data/*", phase='val', img_glob="*.pt")

    loader_kwargs = {
        'batch_size': args.slot_pretrain_batch_size,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': True,
    }

    train_loader = DataLoader(train_dataset, sampler=None, **loader_kwargs)
    val_loader = DataLoader(val_dataset, sampler=None, **loader_kwargs)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    steps = args.slot_train_step
    optimizer_config = {"alg": "Adam",
                        "lr": 0.0004}
    clip_grad_norm = args.slot_clip_grade_norm
    checkpoint_steps = args.slot_save_fre
    logloss_steps = args.slot_log_fre
    logweights_steps = args.slot_log_fre
    logimages_steps = args.slot_log_fre * 10
    logvalid_steps = args.slot_log_fre * 25
    debug = True
    working_dir = args.slot_att_work_path
    dataloaders = [train_loader, val_loader]
    use_exp_decay = args.slot_att_use_exp_decay
    exp_decay_rate = args.slot_att_exp_decay_rate
    exp_decay_steps = args.slot_att_exp_decay_step
    use_warmup_lr: bool = args.slot_att_use_warmup
    warmup_steps = args.slot_att_warmup_step

    trainer = SlotAttentionTrainer(
        device=device,
        steps=steps,
        optimizer_config=optimizer_config,
        clip_grad_norm=None,
        checkpoint_steps=checkpoint_steps,
        logloss_steps=logloss_steps,
        logweights_steps=logweights_steps,
        logimages_steps=logimages_steps,
        logvalid_steps=logvalid_steps,
        debug=debug,
        working_dir=working_dir,
        resubmit_steps=10000000000,
        resubmit_hours=None,
        use_exp_decay=use_exp_decay,
        exp_decay_rate=exp_decay_rate,
        exp_decay_steps=exp_decay_steps,
        use_warmup_lr=use_warmup_lr,
        warmup_steps=warmup_steps,
    )

    trainer.setup(model.to(device), dataloaders)

    trainer.train()
