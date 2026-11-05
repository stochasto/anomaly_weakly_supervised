import pdb
import numpy as np
import torch.utils.data as data
import torch
import utils
from options import *
from config import *
from train import *
from ucf_test import test
from model import *
import os
from dataset_loader import *
from tqdm import tqdm


def create_dataloader(dataset, batch_size, shuffle, num_workers, worker_init_fn=None, drop_last=False):
    """Create a dataloader with safe prefetch handling."""
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        pin_memory=True,
    )

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    return data.DataLoader(dataset, **loader_kwargs)


if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        pdb.set_trace()

    config = Config(args)

    # GPU setup
    gpus = [0]
    torch.cuda.set_device(f'cuda:{gpus[0]}')

    # Disable any seeding -> fully random
    print("[INFO] Running in fully random mode (no seeding).")

    print(f"[INFO] Using feature dimension: 2048")
    print(f"[INFO] Using {config.num_workers} dataloader workers")

    # ---------------- Model ----------------
    config.len_feature = 2048
    net = WSAD(config.len_feature, flag="Train", a_nums=60, n_nums=60).cuda()
    print("[INFO] Model initialized (uncompiled).")

    # ---------------- Dataloaders ----------------
    normal_train_loader = create_dataloader(
        UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal,
                  num_segments=200, len_feature=config.len_feature, is_normal=True),
        batch_size=64, shuffle=True, num_workers=config.num_workers,
        drop_last=True
    )

    abnormal_train_loader = create_dataloader(
        UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal,
                  num_segments=200, len_feature=config.len_feature, is_normal=False),
        batch_size=64, shuffle=True, num_workers=config.num_workers,
        drop_last=True
    )

    test_loader = create_dataloader(
        UCF_crime(root_dir=config.root_dir, mode='Test', modal=config.modal,
                  num_segments=config.num_segments, len_feature=config.len_feature),
        batch_size=1, shuffle=False, num_workers=config.num_workers
    )

    # ---------------- Setup ----------------
    test_info = {"step": [], "auc": [], "ap": [], "ac": []}
    best_auc = 0.0

    criterion = AD_Loss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.lr[0],
        betas=(0.9, 0.999), weight_decay=0.00005
    )

    os.makedirs(args.model_path, exist_ok=True)
    model_file = os.path.join(args.model_path, f"ucf_trans_best.pkl")

    # ---------------- Training Loop ----------------
    print("[INFO] Starting training loop...")
    for step in tqdm(range(1, config.num_iters + 1), total=config.num_iters, dynamic_ncols=True):
        # Adjust learning rate dynamically
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        # Reload dataloader iterators if exhausted
        if (step - 1) % len(normal_train_loader) == 0:
            normal_loader_iter = iter(normal_train_loader)
        if (step - 1) % len(abnormal_train_loader) == 0:
            abnormal_loader_iter = iter(abnormal_train_loader)

        # Train one iteration
        train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion, step)

        # Evaluate periodically — every 5 epochs
        if step % 5 == 0 and step > 1:
            auc_score, ap_score, acc = test(net, config, test_loader, test_info, step)

            # Save AUC-based best model
            if auc_score > best_auc:
                best_auc = auc_score
                utils.save_best_record(
                    test_info,
                    os.path.join(config.output_path, "ucf_best_record.txt")
                )
                torch.save({
                    "step": step,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc": best_auc
                }, model_file)
                print(f"[INFO] ✅ New best AUC: {best_auc:.4f} (checkpoint saved)")

        # Save final model at end
        if step == config.num_iters:
            torch.save(net.state_dict(),
                       os.path.join(args.model_path, f"ucf_trans_final_{step}.pkl"))
            print("[INFO] Training complete. Final model saved.")
