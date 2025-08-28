"""Grid search over (threshold, alpha, update_mode) for tracking.
Usage:
  python scripts/run_ablation.py --config configs/default.yaml \
    --proposals_json data/proposals/det_db_val.json --image_root data/val \
    --ckpt_epoch 80 --output_dir outputs/ablation
"""
import argparse
import os
import yaml

MODES = ["mean", "ema"]
TAUS = [0.2, 0.4, 0.6, 0.8]
ALPHAS = [0.25, 0.5, 0.75, 1.0]


def main(cfg, ckpt_epoch):
    base = cfg.copy()
    base["output_dir"] = os.path.abspath(base.get("output_dir", "outputs/leaf_reid"))

    for tau in TAUS:
        for mode in MODES:
            alphas = [0.0] if mode == "mean" else ALPHAS
            for a in alphas:
                tag = f"tau_{tau}_{mode}_a{a}" if mode == "ema" else f"tau_{tau}_mean"
                cfg_run = base.copy()
                cfg_run.update({
                    "threshold": tau,
                    "update_mode": mode,
                    "alpha": a,
                    "output_dir": os.path.join(base["output_dir"], f"ablation_{tag}"),
                    "epochs": ckpt_epoch,
                })
                # call infer.py
                cmd = (
                    f"python infer.py --config {args.config} "
                    f"--proposals_json {cfg['proposals_json']} --image_root {cfg['image_root']} "
                    f"--output_dir {cfg_run['output_dir']} --threshold {cfg_run['threshold']} "
                    f"--update_mode {cfg_run['update_mode']} --alpha {cfg_run['alpha']}"
                )
                print("\n==>", cmd)
                os.system(cmd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt_epoch", type=int, default=80)
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.ckpt_epoch)