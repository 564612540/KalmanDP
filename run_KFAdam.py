import torch
import math
# from data_utils import generate_Cifar
from KFAdam import KFAdam
from train_utils import train, noisy_train, test
from init_utils import base_parse_args, task_init, logger_init
from fastDP import PrivacyEngine
#PrivacyEngine_Distributed_extending,PrivacyEngine_Distributed_Stage_2_and_3
# from opacus.accountants.utils import get_noise_multiplier
# from opacus.validators import ModuleValidator
import argparse
import warnings
# import timm
# import os
# from datetime import datetime
# import wandb

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='LP DPSGD')
    parser = base_parse_args(parser)
    args = parser.parse_args()
    train_dl, test_dl, model, device, sample_size, acc_step, noise = task_init(args)
    log_file = logger_init(args, noise, sample_size//args.mnbs,type=args.log_type)

    use_manual_noise = not args.clipping and noise>0
    if use_manual_noise:
        noise = noise/args.mnbs
        args.lr = args.lr/acc_step
        print('use manual noise')
    
    optimizer = KFAdam(model.parameters(), lr=args.lr, betas = (0.9, 0.995), weight_decay=0, sigma_dp=noise/args.bs)
    
    # from torch.optim import lr_scheduler
    if args.scheduler:
        from train_utils import CosineAnnealingWarmupRestarts
        lrscheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=args.lr, min_lr=args.lr/10, first_cycle_steps= sample_size//args.bs * args.epoch, warmup_steps= (sample_size*args.epoch)//(args.bs*20))
    else:
        lrscheduler = None
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    if args.clipping:
        privacy_engine = PrivacyEngine(model, noise_multiplier=noise, numerical_stability_constant=1e-3, grad_accum_steps = acc_step, sample_size= sample_size, batch_size=args.bs, epochs= args.epoch, per_sample_clip=args.clipping, torch_seed_is_fixed=False, clipping_fn=args.clipping_fn, clipping_style=args.clipping_style, max_grad_norm=args.clipping_norm)
        privacy_engine.attach(optimizer)

    for E in range(args.epoch):
        # if args.no_record:
        if use_manual_noise:
            # print('using manual noise')
            noisy_train(model, train_dl, optimizer, criterion, log_file, device = device, epoch = E, noise = noise, log_frequency = args.log_freq, acc_step = acc_step,lr_scheduler=lrscheduler)
        else:
            train(model, train_dl, optimizer, criterion, log_file, device = device, epoch = E, log_frequency = args.log_freq, acc_step = acc_step, lr_scheduler=lrscheduler)
        test(model, test_dl, criterion, log_file, device = device, epoch = E)
        
        
