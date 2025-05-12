import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import os
import time

#os.environ["MASTER_ADDR"] = "127.0.0.1"
#os.environ["MASTER_PORT"] = "29505"
#os.environ["RANK"] = "0"
#os.environ["WORLD_SIZE"] = "1"

master_addr = os.environ["MASTER_ADDR"]
master_port = os.environ["MASTER_PORT"]
local_rank = os.environ["LOCAL_RANK"]
rank = os.environ["RANK"]
world_size = os.environ["WORLD_SIZE"]

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print(f"MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")
print(f"LOCAL_RANK: {local_rank}, RANK: {rank}, WORLD_SIZE: {world_size}")


device=torch.device(f"cuda:{rank}")
torch.cuda.set_device(int(rank))
print(f"Current device: {torch.cuda.current_device()} {torch.cuda.get_device_name()} {device}")
x = torch.tensor([1.0], device=f"cuda:{rank}")
print(f"Tensor device: {x.device}")

dist.init_process_group(backend="nccl")
rank=dist.get_rank()
world_size=dist.get_world_size()
print(f"Rank: {rank}, World size: {world_size}")

time.sleep(5)

dist.barrier(device_ids=[device.index])
dist.destroy_process_group()
