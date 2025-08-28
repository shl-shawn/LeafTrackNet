import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models import LeafReIDModel
from datasets import LeafTripletDataset, LeafTripletDatasetV2, LeafTripletDatasetV3
from utils.transforms import get_train_transform
from utils.logging import setup_logger
from utils.dist import set_seed, is_ddp, get_local_rank, is_main_process


def build_dataset(cfg):
    tfm = get_train_transform()
    if cfg["triplet_strategy"].lower() == "v1":
        return LeafTripletDataset(cfg["train_root"], transform=tfm)
    if cfg["triplet_strategy"].lower() == "v2":
        return LeafTripletDatasetV2(cfg["train_root"], transform=tfm)
    if cfg["triplet_strategy"].lower() == "v3":
        return LeafTripletDatasetV3(cfg["train_root"], window_size=cfg["window_size"], transform=tfm)
    raise ValueError("triplet_strategy must be one of [v1, v2, v3]")


def train(cfg):
    set_seed(cfg["seed"])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # model
    model = LeafReIDModel(cfg["backbone"], embed_dim=cfg["embed_dim"], pretrained=True)

    if is_ddp():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = get_local_rank()  # reads LOCAL_RANK env
        torch.cuda.set_device(local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # data
    dataset = build_dataset(cfg)
    if is_ddp():
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["num_workers"], pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True)

    # loss/opt
    criterion = nn.TripletMarginLoss(margin=0.3)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # training loop
    for epoch in range(1, cfg["epochs"] + 1):
        if is_ddp():
            sampler.set_epoch(epoch)
        running, seen = 0.0, 0
        iterator = loader
        if get_local_rank() == 0:
            iterator = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")

        for a, p, n in iterator:
            a, p, n = a.to(device), p.to(device), n.to(device)

            optimizer.zero_grad()
            ea, ep, en = model(a), model(p), model(n)
            loss = criterion(ea, ep, en)
            loss.backward()
            optimizer.step()

            bs = a.size(0)
            running += loss.item() * bs
            seen += bs
            if get_local_rank() == 0:
                iterator.set_postfix({"loss": f"{running/seen:.6f}"})

        if is_main_process() and (epoch % cfg["save_period"] == 0 or epoch == cfg["epochs"]):
            ckpt_dir = os.path.join(cfg["output_dir"], "weights")
            os.makedirs(ckpt_dir, exist_ok=True)
            state = model.module.state_dict() if is_ddp() else model.state_dict()
            torch.save(state, os.path.join(ckpt_dir, f"leaf_reid_e{epoch}.pth"))

    if is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    # override examples
    parser.add_argument("--train_root")
    parser.add_argument("--output_dir")
    parser.add_argument("--backbone")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--triplet_strategy")
    parser.add_argument("--window_size", type=int)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # apply CLI overrides if provided
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v

    train(cfg)