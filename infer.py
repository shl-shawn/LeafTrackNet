import argparse
import os
import yaml
import torch
from PIL import Image
from tqdm import tqdm

from models import LeafReIDModel
from tracking import VisualLeafTracker
from utils.transforms import get_infer_transform
from utils.io import read_detection_json, parse_detection_lines, key_to_plant_frame, write_mot, ensure_dir


def run_inference(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    model = LeafReIDModel(cfg["backbone"], embed_dim=cfg["embed_dim"], pretrained=False).to(device)
    # load weights
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    tracker_args = dict(
        reid_model=model,
        transform=get_infer_transform(),
        device=str(device),
        similarity_threshold=cfg["threshold"],
        max_age=cfg["max_age"],
        update_mode=cfg["update_mode"],
        alpha=cfg["alpha"],
    )

    det_dict = read_detection_json(cfg["proposals_json"])
    # group per-plant
    plants = {}
    for key in det_dict.keys():
        plant, _ = key_to_plant_frame(key)
        plants.setdefault(plant, []).append(key)

    out_root = os.path.join(cfg["output_dir"], "tracks")
    ensure_dir(out_root)

    for plant, keys in tqdm(plants.items(), desc="Plants"):
        keys = sorted(keys, key=lambda k: key_to_plant_frame(k)[1])
        tracker = VisualLeafTracker(**tracker_args)
        lines = []
        for k in keys:
            boxes, _ = parse_detection_lines(det_dict[k])
            plant_id, frame = key_to_plant_frame(k)
            img_path = os.path.join(cfg["image_root"], plant_id, "img", f"{frame:08d}{cfg['image_ext']}")
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            ids, sims = tracker.update(img, boxes)
            for (x1, y1, x2, y2), tid, s in zip(boxes, ids, sims):
                line = f"{frame}, {int(tid)}, {x1:.2f}, {y1:.2f}, {x2-x1:.2f}, {y2-y1:.2f}, {float(s):.4f}, -1, -1, -1"
                lines.append(line)
        write_mot(os.path.join(out_root, f"{plant}.txt"), lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint_path", required=True, help="trained weight needed")
    ap.add_argument("--proposals_json")
    ap.add_argument("--image_root")
    ap.add_argument("--output_dir")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--update_mode")
    ap.add_argument("--alpha", type=float)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    for k, v in vars(args).items():
        if k != "config" and v is not None:
            cfg[k] = v

    run_inference(cfg)