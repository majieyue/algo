import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import os
import time

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29505"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device=torch.device(f'cuda:0')
torch.cuda.set_device(0)
dist.init_process_group(backend="nccl", rank=0, world_size=1)

rank=dist.get_rank()
world_size=dist.get_world_size()
print(f"Rank: {rank}, World size: {world_size}")

time.sleep(5)

dist.barrier(device_ids=[device.index])
dist.destroy_process_group()
