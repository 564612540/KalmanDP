import torch
import math

import torch.distributed
from KFOptimizer import wrap_optimizer
from train_utils_dist import train, test
from init_utils_dist import base_parse_args, task_init, logger_init
from fastDP import PrivacyEngine_Distributed_extending
import deepspeed
import argparse
import warnings
import gc

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='DiSK_Dist')
    parser = base_parse_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    deepspeed.init_distributed()

    if torch.distributed.get_rank() != 0:
        torch.distributed.barrier()

    train_dl, test_dl, model, device, sample_size, acc_step, noise = task_init(args)
    log_file = logger_init(args, noise, sample_size//args.logi_bs,type=args.log_type)

    if torch.distributed.get_rank() == 0:
        torch.distributed.barrier()

    use_manual_noise = not args.clipping and noise>0
    if use_manual_noise:
        RuntimeError("Using manual noise!")

    if args.algo == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = 0)
    elif args.algo == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.algo == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.04)
    # elif args.algo == 'adambc':
    #     optimizer = AdamBC(model.parameters(), lr=args.lr, dp_batch_size=args.bs, dp_l2_norm_clip=1, dp_noise_multiplier=noise, eps_root=1e-8)
    else:
        print(args.algo)
        raise RuntimeError("Unknown Algorithm!")
    
    start = 0
    
    if args.load_path is not None:
        print("loading optimizer")
        checkpoint = torch.load(args.load_path, map_location='cuda')
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
    
    if args.scheduler:
        from train_utils import CosineAnnealingWarmupRestarts
        lrscheduler = CosineAnnealingWarmupRestarts(optimizer, max_lr=args.lr, first_cycle_steps= sample_size//args.logi_bs * args.epoch, warmup_steps= (sample_size*args.epoch)//(args.logi_bs*20), last_epoch = start*sample_size//args.logi_bs-1)
    else:
        lrscheduler = None

    if args.kf:
        optimizer = wrap_optimizer(optimizer, kappa=args.kappa, gamma=args.gamma)
        # optimizer = KFOptimizer(model.parameters(), optimizer=optimizer, kappa=args.kappa, gamma=args.gamma)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    if args.clipping:
        privacy_engine = PrivacyEngine_Distributed_extending(model, noise_multiplier=noise, grad_accum_steps = acc_step, sample_size= sample_size, batch_size=args.logi_bs, epochs= args.epoch, per_sample_clip=args.clipping, torch_seed_is_fixed=True, num_GPUs=torch.distributed.get_world_size())
        # privacy_engine.attach(optimizer)

    model_engine, optimizer, train_dl, lrscheduler = deepspeed.initialize(args=args, model=model, optimizer=optimizer, model_parameters=model.parameters(), training_data=train_dl, lr_scheduler=lrscheduler)

    if args.load_path is not None:
        print("loading optimizer")
        checkpoint = torch.load(args.load_path, map_location='cuda')
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1

    for E in range(start, args.epoch):
        train(model_engine, train_dl, optimizer, criterion, log_file, epoch = E, log_frequency = args.log_freq, acc_step = acc_step, lr_scheduler=lrscheduler)
        print("start testing...",flush=True)
        test(model_engine, test_dl, criterion, log_file, epoch = E)
        gc.collect()
        torch.cuda.empty_cache()
        if args.save_freq >0 and E % args.save_freq == 0 and args.save_path is not None:
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(), 'epoch':E}, args.save_path)
        
        
