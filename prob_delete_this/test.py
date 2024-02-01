import os
import torch

def set_all_seeds(seed): # exclude nondeterminism
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_all_seeds(0)
    print(torch.rand(1))
    world_size = int(os.environ["SLURM_NTASKS"], 1)
    rank = int(os.environ.get("SLURM_PROCID", 0))
    address = os.getenv("SLURM_LAUNCH_NODE_IPADDR")  # Get master node address from SLURM environment variable.
    if address is None:
        raise ValueError("SLURM_LAUNCH_NODE_IPADDR is not set.")  
        
    port = 1234 # Set port number.
    os.environ["MASTER_ADDR"] = str(address) # Set master node address.
    os.environ["MASTER_PORT"] = str(port) # Set master node port.
    
    print(f"world_size: {world_size}, rank: {rank}, address: {address}, port: {port}")
    