import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.optim.lr_scheduler import StepLR


try:
    import comet_ml  # Import before torch
except ImportError:
    pass
from dataset.sg_dataset import get_sketchgraphs_dataloader, get_fsdp_dataloader, SketchGraphsDataset, SketchGraphsCollator
from models.byt5 import ByT5Model
from models.codet5 import CodeT5Model
from models.multimodel import VLModel, train_VL, test_VL

from torch.utils.data import DataLoader
from util import get_loggers, get_checkpoint_callbacks
from args.main_args import get_training_args
from pathlib import Path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from accelerate import Accelerator

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()



def fsdp_main(rank, world_size, tokenizer, args):
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # dataset1 = datasets.MNIST('../data', train=True, download=True,
    #                     transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                     transform=transform)
    
    # train_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="train", shuffle=True)
    # val_dataloader = get_sketchgraphs_dataloader(tokenizer=model.tokenizer, args=args, split="val", shuffle=False)

    # sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    # sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    # train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    # test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    # cuda_kwargs = {'num_workers': 2,
    #                 'pin_memory': True,
    #                 'shuffle': False}
    # train_kwargs.update(cuda_kwargs)
    # test_kwargs.update(cuda_kwargs)

    # train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    train_dataset = SketchGraphsDataset(split='train', args=args)
    val_dataset = SketchGraphsDataset(split='val', args=args)
    collator = SketchGraphsCollator(tokenizer=tokenizer, max_length=args.max_length, args=args)

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)


    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sampler}
    test_kwargs = {'batch_size': args.batch_size, 'sampler': val_sampler}

    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset,**train_kwargs)
    val_loader = DataLoader(val_dataset, **test_kwargs)

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = VLModel(args=args).to(rank)

    model = FSDP(model)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train_VL(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=train_sampler)
        test_VL(model, tokenizer, args, rank, world_size, val_loader)
        scheduler.step()

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if not args.eval:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        # state_dict for FSDP model is only available on Nightlies for now
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()

if __name__ == "__main__":
    fsdp_main()