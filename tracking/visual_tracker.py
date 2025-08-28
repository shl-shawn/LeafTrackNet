import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


class VisualLeafTracker:
    """Cosine-similarity tracker with per-track prototype (mean or EMA)."""
    def __init__(self, reid_model, transform, device="cuda", similarity_threshold=0.7, max_age=2, update_mode="mean", alpha=0.1):
        self.reid_model = reid_model.to(device).eval()
        self.transform = transform
        self.device = device
        self.similarity_threshold = float(similarity_threshold)
        self.max_age = int(max_age)
        self.update_mode = str(update_mode).lower()
        assert self.update_mode in {"mean", "ema"}
        self.alpha = float(alpha)

        self.tracks = []  # dicts with keys: id, emb_sum, emb_count, ema_emb, bbox, age
        self.next_id = 1

    def _embed_patch(self, patch_img):
        with torch.no_grad():
            x = self.transform(patch_img).unsqueeze(0).to(self.device)
            e = self.reid_model(x)
        return e.squeeze(0)

    def _track_proto(self, tr):
        return tr['ema_emb'] if self.update_mode == 'ema' else (tr['emb_sum'] / tr['emb_count'])

    def update(self, pil_image, detections):
        # detections: List[[x1,y1,x2,y2]]
        if len(detections) == 0:
            # age tracks
            kept = []
            for t in self.tracks:
                t['age'] += 1
                if t['age'] <= self.max_age:
                    kept.append(t)
            self.tracks = kept
            return np.array([], dtype=int), np.array([], dtype=float)

        # embed dets
        det_embs = []
        for (x1, y1, x2, y2) in detections:
            patch = pil_image.crop((x1, y1, x2, y2))
            det_embs.append(self._embed_patch(patch))
        det_embs = torch.stack(det_embs, dim=0)
        det_embs = det_embs / det_embs.norm(dim=1, keepdim=True)

        if len(self.tracks) == 0:
            # initialize tracks by descending area to stabilize IDs at t=0
            areas = [ (x2-x1)*(y2-y1) for (x1,y1,x2,y2) in detections ]
            order = np.argsort(areas)[::-1]
            ids = -np.ones(len(detections), dtype=int)
            sims = np.zeros(len(detections), dtype=float)
            for idx in order:
                emb = det_embs[idx].clone()
                self.tracks.append({
                    'id': self.next_id,
                    'emb_sum': emb.clone(),
                    'emb_count': 1,
                    'ema_emb': emb.clone(),
                    'bbox': detections[idx],
                    'age': 0,
                })
                ids[idx] = self.next_id
                sims[idx] = 1.0
                self.next_id += 1
            return ids, sims

        # compute similarity T x N
        protos = torch.stack([ self._track_proto(t) for t in self.tracks ], dim=0)
        protos = protos / protos.norm(dim=1, keepdim=True)
        S = protos @ det_embs.t()  # (T,N)
        cost = (1.0 - S).cpu().numpy()
        t_inds, d_inds = linear_sum_assignment(cost)

        assigned = -np.ones(len(detections), dtype=int)
        sims = np.zeros(len(detections), dtype=float)
        touched = set()

        for ti, di in zip(t_inds, d_inds):
            s = float(S[ti, di].item())
            if s >= self.similarity_threshold:
                assigned[di] = self.tracks[ti]['id']
                sims[di] = s
                touched.add(ti)
                # update prototype
                det_e = det_embs[di]
                if self.update_mode == 'ema':
                    a = self.alpha
                    self.tracks[ti]['ema_emb'] = (1 - a) * self.tracks[ti]['ema_emb'] + a * det_e
                else:
                    self.tracks[ti]['emb_sum'] += det_e
                    self.tracks[ti]['emb_count'] += 1
                self.tracks[ti]['bbox'] = detections[di]
                self.tracks[ti]['age'] = 0

        # new tracks for unmatched detections
        for di in range(len(detections)):
            if assigned[di] == -1:
                emb = det_embs[di].clone()
                self.tracks.append({
                    'id': self.next_id,
                    'emb_sum': emb.clone(),
                    'emb_count': 1,
                    'ema_emb': emb.clone(),
                    'bbox': detections[di],
                    'age': 0,
                })
                assigned[di] = self.next_id
                self.next_id += 1

        # age untouchedb tracks
        new_tracks = []
        for i, t in enumerate(self.tracks):
            if i not in touched:
                t['age'] += 1
            if t['age'] <= self.max_age:
                new_tracks.append(t)
        self.tracks = new_tracks

        return assigned, sims