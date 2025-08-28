import json
import os
from typing import Dict, List, Tuple


def read_detection_json(path: str) -> Dict[str, List[str]]:
    with open(path, 'r') as f:
        return json.load(f)


def parse_detection_lines(lines: List[str]):
    boxes, confs = [], []
    for ln in lines:
        parts = ln.strip().split(',')
        if len(parts) < 5:
            continue
        x, y, w, h, conf = map(float, parts[:5])
        boxes.append([x, y, x + w, y + h])
        confs.append(conf)
    return boxes, confs


def key_to_plant_frame(key: str) -> Tuple[str, int]:
    # Handles keys like "val/Plant-001/00000042.txt" or absolute paths
    base = key.replace('\\', '/')
    parts = base.split('/')
    # find plant id as the element before filename
    if len(parts) < 2:
        raise ValueError(f"Malformed detection key: {key}")
    plant_id = parts[-2]
    frame_str = os.path.splitext(os.path.basename(base))[0]
    frame = int(frame_str)
    return plant_id, frame


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def write_mot(out_txt: str, lines: List[str]):
    ensure_dir(os.path.dirname(out_txt))
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines) + '\n')