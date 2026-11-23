#!/usr/bin/env python3
"""Simulated PushT planning with an ensemble of learned dynamics models."""
from __future__ import annotations
import argparse, copy, csv, hashlib, importlib.metadata, json, math, os, random, time
from math import comb
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gym_pusht.envs.pusht import PushTEnv

TITLE = 'PushT budgeted replanning reference prototype v2'
FIXED = ['fixed1', 'fixed2', 'fixed3', 'fixed4', 'fixed6']
ADAPTIVE = ['error', 'disagreement', 'contact', 'geom', 'random']
METHODS = FIXED + ADAPTIVE
NOMINAL_DAMPING = [0.]                          # training condition: the block stops as soon as it is released
SCALE, ASCALE, ANG_W = 0.02, 0.05, 0.1          # delta normalisation of the dynamics loss
LO, HI = .03125, .96875                         # legal absolute action range, normalised by 512
NEAR_PX = 15.5                                  # near-contact label: signed outline distance of the pusher centre
PAIRS = [('fixed1', 'fixed4'), ('contact', 'random'), ('contact', 'fixed4'), ('error', 'fixed4'), ('geom', 'fixed4')]


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)


def versions():
    return {k: importlib.metadata.version(k) for k in ['gym-pusht', 'gymnasium', 'pymunk', 'torch', 'numpy', 'opencv-python', 'pygame', 'shapely']}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_device(name):
    if name == 'auto': return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    return name


# ---------------------------------------------------------------- environment
def pose6(env):
    return np.array([env.agent.position.x / 512, env.agent.position.y / 512, env.block.position.x / 512, env.block.position.y / 512,
                     math.sin(env.block.angle), math.cos(env.block.angle)], np.float32)


def proximity(env):
    return float(min(s.point_query(env.agent.position).distance for s in env.block.shapes) <= NEAR_PX)


def pose_cost(s, goal):
    return np.sum((s[..., 2:4] - goal[2:4]) ** 2, axis=-1) + .10 * (1 - np.clip(np.sum(s[..., 4:6] * goal[4:6], axis=-1), -1, 1))


def t_sdf(state):
    """Signed distance in pixels from the pusher centre to the T outline, computed from a 6-d pose."""
    d = (state[..., :2] - state[..., 2:4]) * 512; s, c = state[..., 4], state[..., 5]
    x = c * d[..., 0] + s * d[..., 1]; y = -s * d[..., 0] + c * d[..., 1]
    def box(cx, cy, hx, hy):
        qx = np.abs(x - cx) - hx; qy = np.abs(y - cy) - hy
        return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0)
    return np.minimum(box(0, 15, 60, 15), box(0, 75, 15, 45))


class Scene:
    """Reference gym-pusht. Snapshot recipe replays the entire action prefix on the same engine."""
    def __init__(self, seed, damping=0., image_size=64):
        self.seed = int(seed); self.damping = float(damping); self.image_size = int(image_size)
        self.env = PushTEnv(obs_type='state', render_mode='rgb_array', damping=self.damping, observation_width=image_size,
                            observation_height=image_size, visualization_width=256, visualization_height=256)
        self.env.reset(seed=self.seed); rng = np.random.default_rng(self.seed)
        angle = float(rng.uniform(-.5, .5)); origin = np.array([255., 195.]) + rng.uniform(-18, 18, 2)
        self.env.block.angle = angle; self.env.block.position = tuple(origin)
        local = np.array([rng.uniform(-8, 8), 4 * 30 + 24])        # pusher just outside the bottom of the T, object frame
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        self.env.agent.position = tuple(origin + rot @ local)
        self.env.agent.velocity = (0, 0); self.env.block.velocity = (0, 0); self.env.block.angular_velocity = 0
        self.env.space.reindex_shapes_for_body(self.env.block); self.env.space.reindex_shapes_for_body(self.env.agent)
        self.actions = []
    def obs(self): return pose6(self.env)
    def image(self): return self.env._render()
    def step(self, a):
        a = np.clip(np.asarray(a, np.float32), 16, 496); out = self.env.step(a); self.actions.append(a.copy()); return self.obs(), out[1:]
    def recipe(self):
        return {'seed': self.seed, 'damping': self.damping, 'image_size': self.image_size, 'actions': [a.tolist() for a in self.actions]}
    @classmethod
    def restore(cls, recipe):
        scene = cls(recipe['seed'], recipe['damping'], recipe['image_size'])
        for a in recipe['actions']: scene.step(a)
        return scene
    def close(self): self.env.close()


def teacher_action(scene, rng, t):
    """Contact-rich collector that reads simulator state. Used for goal generation and data collection only."""
    s = scene.obs(); p = s[:2] * 512; b = s[2:4] * 512
    towards = b - p; direction = towards / max(np.linalg.norm(towards), 1.); lateral = np.array([-direction[1], direction[0]])
    if t % 12 < 8: a = p + rng.uniform(16, 44) * direction + rng.uniform(-14, 14) * lateral
    else: a = p + rng.uniform(-12, 24) * direction + rng.uniform(-32, 32) * lateral
    return np.clip(a, 16, 496).astype(np.float32)


def explore_action(scene, rng, t, style):
    """Wider coverage than the teacher: fine pushes, reaches around the T, free motion, holds."""
    if style == 0: return teacher_action(scene, rng, t)
    s = scene.obs(); p = s[:2] * 512; b = s[2:4] * 512
    if rng.random() < .08: return np.clip(p, 16, 496).astype(np.float32)
    towards = b - p; direction = towards / max(np.linalg.norm(towards), 1.); lateral = np.array([-direction[1], direction[0]])
    if style == 1: a = p + rng.uniform(-4, 22) * direction + rng.uniform(-12, 12) * lateral
    elif style == 2:
        ang = rng.uniform(0, 2 * math.pi); a = b + rng.uniform(10, 130) * np.array([math.cos(ang), math.sin(ang)])
        if t % 6 >= 3: a = p + rng.normal(0, 30, 2)
    else: a = p + rng.normal(0, 45, 2) if t % 5 else p + rng.uniform(20, 70) * direction
    return np.clip(a, 16, 496).astype(np.float32)


def goal_task(seed, damping, image_size, steps=14):
    """Reachable goal: the block pose the teacher reaches after `steps` actions from the same start."""
    scene = Scene(seed, damping, image_size); rng = np.random.default_rng(seed + 9876)
    for t in range(steps): scene.step(teacher_action(scene, rng, t))
    goal = scene.obs().copy(); scene.close(); start = Scene(seed, damping, image_size)
    start.env.goal_pose = np.array([goal[2] * 512, goal[3] * 512, math.atan2(float(goal[4]), float(goal[5]))])
    return start, goal


# ---------------------------------------------------------------- data
def split_of(e): return 'train' if e % 10 < 7 else 'val' if e % 10 < 9 else 'test'


def pack(out, rows, meta):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    arrays = dict(obs=np.asarray([r['obs'] for r in rows], np.float32), actions=np.asarray([r['actions'] for r in rows], np.float32),
                  contact=np.asarray([r['contact'] for r in rows], np.float32), episode_id=np.asarray([r['id'] for r in rows]),
                  split=np.asarray([r['split'] for r in rows]), damping=np.asarray([r['damping'] for r in rows]))
    arrays['images'] = np.asarray([r['images'] for r in rows], np.uint8) if rows[0]['images'] is not None else np.zeros((0,), np.uint8)
    np.savez_compressed(out / 'dataset.npz', **arrays)
    meta.update(title=TITLE, data_kind='simulated', ownership='reference implementation', episodes=len(rows),
                transitions=int(arrays['actions'].shape[0] * arrays['actions'].shape[1]), has_images=rows[0]['images'] is not None,
                split_counts={s: int(np.sum(arrays['split'] == s)) for s in ['train', 'val', 'test']},
                damping_counts={str(d): int(np.sum(np.isclose(arrays['damping'], d))) for d in sorted(set(arrays['damping'].tolist()))},
                near_contact_rate=float(arrays['contact'].mean()), scale=30, action='absolute target [x,y] in pixels [16,496]',
                success='official gym coverage > 0.95', near_contact=f'signed T-outline distance of pusher centre <= {NEAR_PX} px',
                sha256=sha(out / 'dataset.npz'), versions=versions())
    save_json(out / 'manifest.json', meta)
    print(json.dumps({k: meta[k] for k in ['kind', 'episodes', 'transitions', 'split_counts', 'damping_counts', 'sha256']}, ensure_ascii=False), flush=True)


def collect(a):
    seed_all(a.seed); rows = []; start = time.perf_counter()
    for e in range(a.episodes):
        seed = a.seed * 100000 + e; rng = np.random.default_rng(seed + 31415)
        d = a.dampings[int(rng.integers(len(a.dampings)))]; style = int(rng.choice([0, 1, 1, 2, 3]))    # independent of each other and of the split
        scene = Scene(seed, d, a.image_size)
        for t in range(int(rng.integers(0, 15))): scene.step(teacher_action(scene, rng, t))               # varied block poses
        r = dict(obs=[scene.obs()], images=[scene.image()] if a.images else None, actions=[], contact=[], id=seed, split=split_of(e), damping=d)
        for t in range(a.steps):
            act = explore_action(scene, rng, t, style); scene.step(act)
            r['actions'].append(act / 512); r['contact'].append(proximity(scene.env)); r['obs'].append(scene.obs())
            if a.images: r['images'].append(scene.image())
        rows.append(r); scene.close()
        if (e + 1) % 400 == 0 or e + 1 == a.episodes: print(f'episodes {e + 1}/{a.episodes}', flush=True)
    pack(a.out, rows, dict(kind='exploration', seed=a.seed, steps=a.steps, image_size=a.image_size, seconds=time.perf_counter() - start))


def load_data(paths, need_images=False, dampings=None):
    parts = [np.load(p, allow_pickle=False) for p in paths]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ['obs', 'actions', 'contact', 'split', 'damping', 'episode_id']}
    if need_images:
        if any(p['images'].ndim < 5 for p in parts): raise ValueError('pixels mode needs datasets collected with --images')
        d['images'] = np.concatenate([p['images'] for p in parts])
    if dampings is not None:
        keep = np.isin(np.round(d['damping'], 3), np.round(dampings, 3)); d = {k: v[keep] for k, v in d.items()}
    return d


def rows_of(states, actions, contacts, targets):
    """Model rows: previous pose, current pose, action, previously commanded action, next true pose, next near-contact label."""
    prev = np.concatenate([states[:, :1], states[:, :-2]], axis=1)
    pa = np.concatenate([states[:, :1, :2], actions[:, :-1]], axis=1)          # "hold" at the first step of an episode
    flat = lambda x: torch.from_numpy(np.ascontiguousarray(x.reshape(-1, x.shape[-1]))).float()
    return [flat(prev), flat(states[:, :-1]), flat(actions), flat(pa), flat(targets[:, 1:]), torch.from_numpy(contacts.reshape(-1).copy()).float()]


# ---------------------------------------------------------------- models
def mlp(i, h, o, layers=3):
    mods = []; d = i
    for _ in range(layers): mods += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*mods, nn.Linear(d, o))


class Dynamics(nn.Module):
    """Inputs: previous pose, current pose, action, previous action. Block-frame features and deltas; separate near-contact trunk."""
    def __init__(self, hidden=256):
        super().__init__(); self.pose = mlp(21, hidden, 5); self.contact = mlp(21, 96, 1, 2)
    @staticmethod
    def feats(prev, now, action, pa):
        s, c = now[:, 4:5], now[:, 5:6]
        rot = lambda v: torch.cat([c * v[:, 0:1] + s * v[:, 1:2], -s * v[:, 0:1] + c * v[:, 1:2]], -1)
        vel = now[:, :4] - prev[:, :4]; dth = now[:, 4:5] * prev[:, 5:6] - now[:, 5:6] * prev[:, 4:5]
        def sdf(v):         # signed distance to the known T outline, block frame, in units of 64 px
            x = v[:, 0:1] * 512; y = v[:, 1:2] * 512
            def box(cx, cy, hx, hy):
                qx = (x - cx).abs() - hx; qy = (y - cy).abs() - hy
                return torch.sqrt(qx.clamp(min=0) ** 2 + qy.clamp(min=0) ** 2 + 1e-9) + torch.maximum(qx, qy).clamp(max=0)
            return torch.minimum(box(0, 15, 60, 15), box(0, 75, 15, 45)) / 64
        return torch.cat([now, rot(now[:, :2] - now[:, 2:4]) * 4, rot(action - now[:, 2:4]) * 4, rot(action - now[:, :2]) * 8,
                          rot(vel[:, :2]) * 40, rot(vel[:, 2:4]) * 40, dth * 20, rot(pa - prev[:, :2]) * 8,
                          sdf(rot(now[:, :2] - now[:, 2:4])), sdf(rot(action - now[:, 2:4]))], -1)
    def forward(self, prev, now, action, pa):
        f = self.feats(prev, now, action, pa); y = self.pose(f); s, c = now[:, 4:5], now[:, 5:6]
        back = lambda v: torch.cat([c * v[:, 0:1] - s * v[:, 1:2], s * v[:, 0:1] + c * v[:, 1:2]], -1)
        p = now[:, :2] + back(y[:, 0:2]) * SCALE; b = now[:, 2:4] + back(y[:, 2:4]) * SCALE; d = y[:, 4:5] * ASCALE
        return torch.cat([p.clamp(0, 1), b.clamp(0, 1), s * torch.cos(d) + c * torch.sin(d), c * torch.cos(d) - s * torch.sin(d)], -1), self.contact(f)[:, 0]


class Vision(nn.Module):
    """Image to pose. Keeps an 8x8 spatial map and two coordinate channels."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(5, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(96, 96, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(), nn.Linear(96 * 64, 256), nn.ReLU(), nn.Linear(256, 6))
    def forward(self, x):
        n, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(n, 1, h, w); xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(n, 1, h, w)
        y = self.net(torch.cat([x, xs, ys], 1))
        return torch.cat([y[:, :4], F.normalize(y[:, 4:6], dim=-1, eps=1e-6)], -1)


def image_tensor(images, device='cpu'):
    """RGB-only preprocessing shared by training and deployment: the green goal overlay is painted white."""
    pixels = np.asarray(images).copy(); colors = pixels.astype(np.int16)
    green = (colors[..., 1] > colors[..., 0] + 20) & (colors[..., 1] > colors[..., 2] + 20); pixels[green] = 255
    return torch.from_numpy(pixels).to(device).float().permute(0, 3, 1, 2) / 255.


def estimate(vision, images, device, batch=512):
    flat = images.reshape(-1, *images.shape[-3:]); out = []
    with torch.no_grad():
        for i in range(0, len(flat), batch): out.append(vision(image_tensor(flat[i:i + batch], device)).cpu().numpy())
    return np.concatenate(out).reshape(*images.shape[:2], 6)


def pose_loss(pred, target):
    return (((pred[:, :4] - target[:, :4]) / SCALE) ** 2).mean() + ANG_W * (((pred[:, 4:6] - target[:, 4:6]) / ASCALE) ** 2).mean()


def train(a):
    seed_all(a.seed); torch.set_num_threads(6); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = choose_device(a.device)
    d = load_data(a.data, a.mode == 'pixels', a.dampings); spl = d['split']; states = d['obs']; model_states = states; vision_state = None; vlog = []
    if not all(np.any(spl == x) for x in ['train', 'val', 'test']): raise ValueError('collect at least 10 episodes for episode-disjoint train/val/test splits')
    if a.mode == 'pixels':      # only training episodes supervise the image-to-pose front-end
        vision = Vision().to(dev); opt = torch.optim.Adam(vision.parameters(), lr=1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.vision_epochs)
        tri = d['images'][spl == 'train'].reshape(-1, *d['images'].shape[-3:]); tr_y = states[spl == 'train'].reshape(-1, 6)
        vai = d['images'][spl == 'val']; vay = states[spl == 'val']; best = float('inf'); rng = np.random.default_rng(a.seed)
        for ep in range(a.vision_epochs):
            vision.train(); perm = rng.permutation(len(tr_y))
            for i in range(0, len(perm), 256):
                idx = np.sort(perm[i:i + 256]); pred = vision(image_tensor(tri[idx], dev)); tgt = torch.from_numpy(tr_y[idx]).to(dev)
                loss = F.mse_loss(pred[:, :4], tgt[:, :4]) + .02 * F.mse_loss(pred[:, 4:6], tgt[:, 4:6])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step(); vision.eval(); est = estimate(vision, vai, dev)
            rmse = float(np.sqrt(np.mean((est[..., :4] - vay[..., :4]) ** 2)) * 512)
            ang = float(np.degrees(np.arccos(np.clip(np.sum(est[..., 4:6] * vay[..., 4:6], -1), -1, 1))).mean())
            vlog.append(dict(epoch=ep + 1, val_pos_rmse_px=rmse, val_angle_mae_deg=ang))
            if rmse + ang < best: best = rmse + ang; vision_state = {k: v.detach().cpu().clone() for k, v in vision.state_dict().items()}
            print(f'vision epoch {ep + 1}: val rmse {rmse:.2f}px angle {ang:.2f}deg', flush=True)
        vision.load_state_dict(vision_state); vision.eval(); model_states = estimate(vision, d['images'], dev)
    # Dynamics always predicts simulator pose labels; in pixels mode its inputs are image-estimated poses.
    tr = rows_of(model_states[spl == 'train'], d['actions'][spl == 'train'], d['contact'][spl == 'train'], states[spl == 'train'])
    va = rows_of(model_states[spl == 'val'], d['actions'][spl == 'val'], d['contact'][spl == 'val'], states[spl == 'val'])
    members = []; logs = []
    for m in range(a.members):
        torch.manual_seed(a.seed + 101 * m); net = Dynamics(a.hidden); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); rng = np.random.default_rng(a.seed + 101 * m)
        sample = torch.from_numpy(rng.integers(0, len(tr[0]), len(tr[0]))); best_p = best_c = float('inf'); keep = {}
        for ep in range(a.epochs):
            net.train(); perm = sample[torch.randperm(len(sample))]
            for i in range(0, len(perm), a.batch):
                idx = perm[i:i + a.batch]; pred, c = net(tr[0][idx], tr[1][idx], tr[2][idx], tr[3][idx])
                loss = pose_loss(pred, tr[4][idx]) + F.binary_cross_entropy_with_logits(c, tr[5][idx])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.); opt.step()
            sched.step(); net.eval()
            with torch.no_grad(): pred, c = net(*va[:4]); vp = float(pose_loss(pred, va[4])); vc = float(F.binary_cross_entropy_with_logits(c, va[5]))
            logs.append(dict(member=m, epoch=ep + 1, val_pose_loss=vp, val_contact_bce=vc)); sd = net.state_dict()
            # the pose trunk and the near-contact trunk share no parameters; each keeps its own best validation epoch
            if vp < best_p: best_p = vp; keep.update({k: v.clone() for k, v in sd.items() if k.startswith('pose')})
            if vc < best_c: best_c = vc; keep.update({k: v.clone() for k, v in sd.items() if k.startswith('contact')})
        members.append(keep); print(f'member {m}: val_pose_loss={best_p:.5f} val_contact_bce={best_c:.4f}', flush=True)
    torch.save(dict(format='v2', mode=a.mode, members=members, vision=vision_state, hidden=a.hidden, seed=a.seed, image_size=a.image_size,
                    train_dampings=sorted(set(np.round(d['damping'], 3).tolist())), preprocess_version='green_overlay_removed_v1' if a.mode == 'pixels' else 'not_applicable',
                    data_sha256=[sha(p) for p in a.data], training_source_sha256=sha(__file__), args=vars(a)), out / 'model.pt')
    save_json(out / 'training_log.json', dict(dynamics=logs, vision=vlog))
    summary = dict(mode=a.mode, members=a.members, train_transitions=len(tr[0]), validation_transitions=len(va[0]), test_episodes_untouched=int(np.sum(spl == 'test')),
                   train_dampings=sorted(set(np.round(d['damping'], 3).tolist())), parameters_per_dynamics=sum(p.numel() for p in Dynamics(a.hidden).parameters()),
                   vision_parameters=0 if vision_state is None else sum(v.numel() for v in vision_state.values()), vision_device=dev, data=[str(p) for p in a.data], versions=versions())
    save_json(out / 'training_summary.json', summary); print(json.dumps({k: summary[k] for k in ['mode', 'train_transitions', 'validation_transitions', 'train_dampings', 'parameters_per_dynamics', 'vision_parameters']}), flush=True)


# ---------------------------------------------------------------- planning
class Budget:
    def __init__(self, limit): self.limit = int(limit); self.used = 0
    @property
    def remaining(self): return self.limit - self.used
    def charge(self, n):
        if n < 0 or n > self.remaining: raise RuntimeError('prediction budget exceeded')
        self.used += int(n)


def cog(state):
    """Geometric centre of the T. The body origin is the middle of the top bar."""
    return state[..., 2:4] + np.stack([-state[..., 4], state[..., 5]], -1) * (45 / 512)


def behind_point(cur, goal, gap=95 / 512):
    """Where a push towards the goal can start: opposite the goal direction, just clear of the outline."""
    c = cog(cur); d = cog(goal) - c; dist = float(np.linalg.norm(d)); dhat = d / max(dist, 1e-6)
    return c, dhat, dist, c - dhat * gap


def propose(rng, warm, cur, goal, K, H):
    p = cur[:2]; c, dhat, dist, q = behind_point(cur, goal)
    stride = float(np.clip(dist / 2, .006, .055))                    # stride shrinks with the remaining distance
    straight = np.stack([p + dhat * stride * (h + 1) for h in range(H)])
    seqs = [straight, np.repeat(p[None], H, 0), np.repeat(q[None], H, 0)]
    rel = p - c; rad = max(float(np.linalg.norm(rel)), 95 / 512); rel = rel / max(float(np.linalg.norm(rel)), 1e-6) * rad
    for deg in (-70, -35, 35, 70):                                   # orbit around the T without touching it
        t = math.radians(deg); seqs.append(np.repeat((c + np.array([math.cos(t) * rel[0] - math.sin(t) * rel[1], math.sin(t) * rel[0] + math.cos(t) * rel[1]]))[None], H, 0))
    if warm is not None: seqs.append(warm)                           # remaining part of the cached plan
    seqs = seqs[:max(2, K // 2)]; n = K - len(seqs); n_reach = n // 4; n_noise = n - n_reach
    sig = .045 * 2. ** (-rng.uniform(0, 4, n_noise))                 # multi-scale noise: about 23 px down to 1.4 px
    noise = np.cumsum(rng.normal(0, 1, (n_noise, H, 2)) * sig[:, None, None], 1) / np.sqrt(np.arange(1, H + 1))[None, :, None]
    centre = np.stack([straight if (warm is None or i % 2) else warm for i in range(n_noise)])
    ang = rng.uniform(0, 2 * math.pi, n_reach); r = rng.uniform(.02, .26, n_reach)      # constant reach targets around the T
    reach = np.repeat((cur[2:4][None] + r[:, None] * np.stack([np.cos(ang), np.sin(ang)], 1))[:, None], H, 1)
    return np.clip(np.concatenate([np.stack(seqs), centre + noise, reach]), LO, HI)


class Agent:
    def __init__(self, path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck.get('format') != 'v2':
            raise RuntimeError('Checkpoint format mismatch: this pipeline requires a v2 checkpoint. '
                               'From the repository root, regenerate compatible models with '
                               '`python run_protocol.py --profile pilot --root rebuilt --prefix rebuilt_ --execute`. '
                               'Then use rebuilt/models/rebuilt_state_pilot/model.pt or rebuilt/models/rebuilt_pixels_pilot/model.pt.')
        self.mode = ck['mode']; self.image_size = ck['image_size']; self.model_sha256 = sha(path); self.preprocess_version = ck['preprocess_version']
        self.models = [Dynamics(ck['hidden']).eval() for _ in ck['members']]
        for n, w in zip(self.models, ck['members']): n.load_state_dict(w)
        self.vision = None
        if self.mode == 'pixels': self.vision = Vision().eval(); self.vision.load_state_dict(ck['vision'])
        cal = Path(path).parent / 'calibration.json'; self.cal = json.loads(cal.read_text()) if cal.exists() else None
        self.vision_calls = 0; self.E = len(self.models)
    def observe(self, scene):
        if self.mode == 'state': return scene.obs()
        with torch.no_grad(): s = self.vision(image_tensor(scene.image()[None]))[0].numpy()
        self.vision_calls += 1; return s
    def rollout(self, prev, cur, action, last):
        K, H = action.shape[:2]; at = torch.from_numpy(action.astype(np.float32)); preds = []; cons = []; l0 = torch.from_numpy(np.repeat(last[None].astype(np.float32), K, 0))
        with torch.no_grad():
            for net in self.models:
                pv = torch.from_numpy(np.repeat(prev[None], K, 0)); nw = torch.from_numpy(np.repeat(cur[None], K, 0)); ps = []; cs = []
                for h in range(H):
                    nx, c = net(pv, nw, at[:, h], at[:, h - 1] if h else l0); ps.append(nx); cs.append(c.sigmoid()); pv, nw = nw, nx
                preds.append(torch.stack(ps, 1).numpy()); cons.append(torch.stack(cs, 1).numpy())
        pred = np.stack(preds); return pred.mean(0), (pred[..., :4].std(0) ** 2).mean(-1) ** .5, np.stack(cons).mean(0)
    def plan(self, prev, cur, goal, budget, rng, last, horizon=6, candidates=32, iters=2, warm=None, tail=6, shape=.3):
        """`horizon` free actions, then `tail` steps repeating the last target so motion after release is predicted and charged."""
        T = horizon + tail; K = min(candidates, budget.remaining // (self.E * T * iters))
        if K < 4: return None
        action = propose(rng, warm, cur, goal, K, horizon); q = behind_point(cur, goal)[3]; now_cost = float(pose_cost(cur, goal))
        for it in range(iters):
            budget.charge(self.E * K * T); full = np.concatenate([action, np.repeat(action[:, -1:], tail, 1)], 1)
            mean, std, contact = self.rollout(prev, cur, full, last)
            costs = pose_cost(mean[:, -1], goal) + .20 * pose_cost(mean, goal).mean(1)        # risk weight on disagreement is zero in this version
            costs = costs + shape * now_cost * np.sum((mean[:, horizon - 1, :2] - q) ** 2, -1)  # tie-breaker: end where a useful push can start
            if it + 1 < iters:      # one refinement round around the best candidates
                elite = action[np.argsort(costs)[:max(4, K // 6)]]; mu = elite.mean(0); sd = elite.std(0) + .002
                action = np.clip(np.concatenate([elite[:1], mu[None] + rng.normal(0, 1, (K - 1, horizon, 2)) * sd[None]]), LO, HI)
        i = int(np.argmin(costs))
        return dict(actions=action[i] * 512, predictions=mean[i], disagreement=std[i], contact=contact[i], K=K, cost=float(costs[i]), near_at_plan=bool(t_sdf(cur) <= NEAR_PX + 2))


class OracleAgent(Agent):
    """Same proposals and cost, candidates scored by replaying the real simulator. Offline planner ceiling, never a deployable controller."""
    def __init__(self): self.mode = 'oracle'; self.image_size = 64; self.E = 1; self.vision_calls = 0; self.scene = None; self.cal = None; self.model_sha256 = None; self.preprocess_version = 'not_applicable'
    def observe(self, scene): self.scene = scene; return scene.obs()
    def rollout(self, prev, cur, action, last=None):
        recipe = self.scene.recipe(); out = []
        for seq in action:
            s = Scene.restore(recipe); out.append([s.step(act * 512)[0] for act in seq]); s.close()
        m = np.asarray(out, np.float32); return m, np.zeros(m.shape[:2], np.float32), np.zeros(m.shape[:2], np.float32)


# ---------------------------------------------------------------- triggers
def trigger_diagnostics(method, age, plan, cur, cal, max_commit, rng):
    """All monitoring reuses the cached plan; no extra dynamics call. Scores are logged for every method."""
    if plan is None: return dict(trigger=True, reason='no_cached_plan', error=None, disagreement=None, contact=None, geom=None)
    n = len(plan['predictions']); idx = min(max(0, age - 1), n - 1); j = min(age, n - 1); lo = max(0, j - 1); hi = min(n, j + 2)
    info = dict(trigger=False, reason='reuse_cached_plan', error=float(np.sqrt(np.mean((cur[:4] - plan['predictions'][idx, :4]) ** 2))), disagreement=float(plan['disagreement'][j]),
                contact=float(np.ptp(plan['contact'][lo:hi])), geom=float(bool(t_sdf(cur) <= NEAR_PX + 2) != plan['near_at_plan']))
    if method.startswith('fixed'):
        if age >= int(method[5:]): info.update(trigger=True, reason='periodic')
    elif age >= max_commit: info.update(trigger=True, reason='max_commit')
    elif method == 'geom':
        if info['geom'] > .5: info.update(trigger=True, reason='geometry_event')
    elif method == 'random':
        if rng.random() < cal['random_rate']: info.update(trigger=True, reason='random')
    elif method in ('error', 'disagreement', 'contact'):
        if info[method] > cal[method + '_threshold']: info.update(trigger=True, reason=method + '_threshold')
    else: raise ValueError(method)
    return info


# ---------------------------------------------------------------- closed loop
def run_episode(agent, task, seed, damping, method, a, cal, explore=None, record_video=False):
    scene, goal = goal_task(seed, damping, agent.image_size, a.goal_steps); budget = Budget(a.budget if a.budget > 0 else 10 ** 12)
    # The goal pose is a legitimate task specification in both modes; the future trajectory is never read by the controller.
    rng = np.random.default_rng(seed); trng = np.random.default_rng(seed + 77); before_vision = agent.vision_calls
    prev = agent.observe(scene); cur = prev.copy(); last = cur[:2].copy(); plan = None; age = 0; replans = 0; exhausted = False; status = 'running'; lat = []
    truth = scene.obs(); initial = float(pose_cost(truth, goal)); best_cov = float(scene.env._get_coverage()); best_cost = initial; hold = 0; nonper = 0; eligible = 0
    traces = []; data = dict(obs=[truth], images=[scene.image()] if explore and explore['images'] else None, actions=[], contact=[]); video = []
    for t in range(a.steps):
        t0 = time.perf_counter(); age_before = age; budget_before = budget.used; diag = trigger_diagnostics(method, age, plan, cur, cal, a.max_commit, trng); did = 0
        if plan is not None and method in ADAPTIVE and age < a.max_commit: eligible += 1; nonper += int(diag['trigger'])
        if diag['trigger'] and not exhausted:
            warm = None
            if plan is not None and age < len(plan['actions']):
                rest = plan['actions'][age:] / 512; warm = np.concatenate([rest, np.repeat(rest[-1:], a.horizon - len(rest), 0)])
            new = agent.plan(prev, cur, goal, budget, rng, last, a.horizon, a.candidates, a.cem_iters, warm, a.tail, a.shape)
            if new is not None: plan = new; age = 0; replans += 1; did = 1
            else:
                exhausted = True; status = 'budget_exhausted'
                if a.exhaust == 'hold': plan = None
        if plan is None or age >= len(plan['actions']): act = cur[:2] * 512; hold += 1      # hold the latest estimated pusher position
        else: act = plan['actions'][age]
        if explore and explore['rng'].random() < explore['eps']: act = act + explore['rng'].normal(0, explore['sigma'], 2)
        before = cur.copy(); scene.step(act); last = (np.clip(act, 16, 496) / 512).astype(np.float32); prev, cur = before, agent.observe(scene); age += 1; lat.append(time.perf_counter() - t0)
        truth = scene.obs(); cov = float(scene.env._get_coverage()); tc = float(pose_cost(truth, goal)); best_cov = max(best_cov, cov); best_cost = min(best_cost, tc)
        if explore:
            data['actions'].append(np.clip(act, 16, 496) / 512); data['contact'].append(proximity(scene.env)); data['obs'].append(truth)
            if data['images'] is not None: data['images'].append(scene.image())
        traces.append(dict(task=task, seed=seed, method=method, t=t, coverage=cov, true_cost=tc, estimated_cost=float(pose_cost(cur, goal)), near_contact=proximity(scene.env), status=status,
                           age_before=age_before, budget_before=budget_before, budget_after=budget.used, replanned=did, triggered=int(diag['trigger']), trigger_reason=diag['reason'],
                           score_error=diag['error'], score_disagreement=diag['disagreement'], score_contact=diag['contact'], score_geom=diag['geom']))
        if record_video: video.append(scene.env.render())
        if cov > .95 and not explore: status = 'success'; break
    row = dict(task=task, seed=seed, damping=damping, method=method, mode=agent.mode, success=int(status == 'success'), steps=t + 1, coverage=cov, max_coverage=best_cov,
               initial_true_cost=initial, final_true_cost=tc, min_true_cost=best_cost, progress=1 - tc / initial, position_error_px=float(np.linalg.norm(truth[2:4] - goal[2:4]) * 512),
               angle_error_deg=math.degrees(math.acos(float(np.clip(np.dot(truth[4:6], goal[4:6]), -1, 1)))), model_calls=budget.used, budget=a.budget, replans=replans, hold_steps=hold,
               nonperiodic_triggers=nonper, eligible_steps=eligible, vision_calls=agent.vision_calls - before_vision, mean_step_ms=1000 * float(np.mean(lat)), p95_step_ms=1000 * float(np.quantile(lat, .95)), status=status)
    scene.close(); return row, traces, data, video


def summarise(rows, methods):
    out = {}
    for m in methods:
        r = [x for x in rows if x['method'] == m]; f = lambda k: float(np.mean([x[k] for x in r]))
        out[m] = dict(successes=sum(x['success'] for x in r), tasks=len(r), mean_max_coverage=f('max_coverage'), mean_final_coverage=f('coverage'), median_progress=float(np.median([x['progress'] for x in r])),
                      ended_worse_than_start=sum(x['progress'] < 0 for x in r), mean_model_calls=f('model_calls'), mean_replans=f('replans'), mean_hold_steps=f('hold_steps'),
                      budget_exhausted=sum(x['status'] == 'budget_exhausted' for x in r), nonperiodic_trigger_rate=sum(x['nonperiodic_triggers'] for x in r) / max(1, sum(x['eligible_steps'] for x in r)), mean_step_ms=f('mean_step_ms'))
    return out


def identical_groups(traces, methods):
    key = {m: [(x['task'], x['t'], round(x['coverage'], 9), x['budget_after']) for x in traces if x['method'] == m] for m in methods}; groups = []
    for m in methods:
        for g in groups:
            if key[g[0]] == key[m]: g.append(m); break
        else: groups.append([m])
    return groups


def write_run(out, rows, traces, methods, extra):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, data in [('episodes.csv', rows), ('traces.csv', traces)]:
        with open(out / name, 'w', newline='') as f: w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summary = dict(kind='simulated reference pilot', goal_input='specified target block pose; current pose comes from the image CNN in pixels mode', methods=summarise(rows, methods),
                   identical_behaviour_groups=identical_groups(traces, methods), versions=versions(),
                   latency_scope='environment-step loop including trigger check, planning, simulator step and current observation; CPU inference; noisier with --workers > 1',
                   limitations=['single training seed unless explicitly repeated', 'default T scale 30 only', 'model trained at damping 0 only; other dampings are unseen dynamics',
                                'goals are generated separately under each damping, so cross-damping differences are not fixed-goal robustness effects', 'success definition differs from swm/PushT-v1'], **extra)
    save_json(out / 'summary.json', summary)
    for m, s in summary['methods'].items():
        print(f"{m:13s} success {s['successes']:2d}/{s['tasks']} max_cov {s['mean_max_coverage']:.3f} median_progress {s['median_progress']:+.3f} calls {s['mean_model_calls']:7.0f} replans {s['mean_replans']:5.1f} exhausted {s['budget_exhausted']:2d} trigger_rate {s['nonperiodic_trigger_rate']:.2f}")
    print('identical behaviour groups:', summary['identical_behaviour_groups'], flush=True)


_W = {}


def _init_worker(model):
    torch.set_num_threads(1); _W['agent'] = Agent(model) if model else OracleAgent()


def _eval_task(job):
    a, task, methods, cal = job; agent = _W['agent']; seed = 900000 + a.seed * 1000 + task; rows = []; traces = []
    for m in methods:
        row, tr, _, video = run_episode(agent, task, seed, a.damping, m, a, cal, record_video=task < a.gif); rows.append(row); traces += tr
        if video:
            import imageio.v2 as imageio
            Path(a.out).mkdir(parents=True, exist_ok=True); imageio.mimsave(Path(a.out) / f'task{task}_{m}.gif', video, duration=100, loop=0)
    return rows, traces


def _pool(a, model, fn, jobs):
    if a.workers <= 1: _init_worker(model); return [fn(j) for j in jobs]
    with Pool(a.workers, initializer=_init_worker, initargs=(model,)) as pool: return pool.map(fn, jobs)


def evaluate(a):
    seed_all(a.seed); methods = a.methods.split(','); start = time.perf_counter(); cal = Agent(a.model).cal
    if not set(methods) <= set(METHODS): raise ValueError('unknown method')
    if cal is None and set(methods) & set(ADAPTIVE): raise RuntimeError('No calibration.json beside the model; run the calibrate command first or pass only fixed methods.')
    res = _pool(a, a.model, _eval_task, [(a, t, methods, cal) for t in range(a.tasks)]); agent = Agent(a.model)
    write_run(a.out, [r for x in res for r in x[0]], [t for x in res for t in x[1]], methods,
              dict(mode=agent.mode, model_sha256=agent.model_sha256, preprocess_version=agent.preprocess_version, args=vars(a), calibration=cal,
                   plan_cost_units=agent.E * a.candidates * (a.horizon + a.tail) * a.cem_iters, seconds=time.perf_counter() - start))


def oracle(a):
    a.gif = 0; a.model = None; start = time.perf_counter(); methods = a.methods.split(',')
    res = _pool(a, None, _eval_task, [(a, t, methods, {}) for t in range(a.tasks)])
    write_run(a.out, [r for x in res for r in x[0]], [t for x in res for t in x[1]], methods, dict(mode='oracle dynamics: planner ceiling only', args=vars(a), seconds=time.perf_counter() - start))


def _policy_task(job):
    a, e, cfg = job; seed = 700000 + a.seed * 10000 + e; ex = dict(rng=np.random.default_rng(seed + 5), eps=a.eps, sigma=a.sigma, images=a.images)
    d = a.dampings[int(ex['rng'].integers(len(a.dampings)))]; row, _, data, _ = run_episode(_W['agent'], e, seed, d, str(ex['rng'].choice(['fixed1', 'fixed2', 'fixed4'])), cfg, {}, explore=ex)
    return dict(obs=data['obs'], images=data['images'], actions=data['actions'], contact=data['contact'], id=seed, split=split_of(e), damping=d), int(row['max_coverage'] > .95)


def collect_policy(a):
    """On-policy aggregation: run the current controller on training task seeds at training dampings and keep the true transitions."""
    seed_all(a.seed); start = time.perf_counter()
    cfg = argparse.Namespace(goal_steps=14, budget=0, steps=a.steps, horizon=6, candidates=32, cem_iters=2, max_commit=6, exhaust='suffix', tail=a.tail, shape=a.shape)
    res = _pool(a, a.model, _policy_task, [(a, e, cfg) for e in range(a.episodes)])
    pack(a.out, [r[0] for r in res], dict(kind='on-policy aggregation', seed=a.seed, steps=a.steps, image_size=64, model=str(a.model), model_sha256=sha(a.model), action_noise=dict(probability=a.eps, sigma_px=a.sigma),
                                         episodes_reaching_coverage_095=sum(r[1] for r in res), seconds=time.perf_counter() - start))


def calibrate(a):
    """Closed-loop trigger calibration on validation task seeds at training dampings only."""
    seed_all(a.seed); agent = Agent(a.model); cfg = argparse.Namespace(goal_steps=14, budget=0, steps=35, horizon=6, candidates=32, cem_iters=2, max_commit=a.max_commit, exhaust='suffix', tail=a.tail, shape=a.shape)
    def sweep(method, cal):
        rows = []; tr = []
        for task in range(a.tasks):
            r, t, _, _ = run_episode(agent, task, 900000 + a.seed * 1000 + task, a.dampings[task % len(a.dampings)], method, cfg, cal); rows.append(r); tr += t
        return rows, tr
    _, tr = sweep(f'fixed{a.max_commit}', {})
    cal = dict(target_rate=a.rate, max_commit=a.max_commit, validation_tasks=a.tasks, task_seed=a.seed, dampings=a.dampings,
               source='closed-loop validation rollouts under a fixed max-commit schedule; empirical thresholds, no coverage guarantee')
    for k in ['error', 'disagreement', 'contact']:
        v = np.array([x['score_' + k] for x in tr if x['score_' + k] is not None]); cal[k + '_threshold'] = float(np.quantile(v, 1 - a.rate)); cal[k + '_score_quantiles_10_50_90_99'] = [float(q) for q in np.quantile(v, [.1, .5, .9, .99])]
    cal['random_rate'] = 0.; cal['realised_rate'] = {}
    for m in ['error', 'disagreement', 'contact', 'geom']:
        rows, _ = sweep(m, cal); cal['realised_rate'][m] = sum(r['nonperiodic_triggers'] for r in rows) / max(1, sum(r['eligible_steps'] for r in rows))
    cal['random_rate'] = cal['realised_rate']['contact']          # the random control is matched to the contact rule's realised rate
    save_json(Path(a.model).parent / 'calibration.json', cal); print(json.dumps({k: cal[k] for k in ['error_threshold', 'disagreement_threshold', 'contact_threshold', 'random_rate', 'realised_rate']}), flush=True)


# ---------------------------------------------------------------- audit, report, gate
def auroc(score, label):
    order = np.argsort(score); rank = np.empty(len(score)); rank[order] = np.arange(1, len(score) + 1); pos = label > .5; n1 = pos.sum(); n0 = len(label) - n1
    return float((rank[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def audit(a):
    seed_all(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); scene = Scene(7123, 0.); rng = np.random.default_rng(182)
    for t in range(5): scene.step(teacher_action(scene, rng, t))
    recipe = scene.recipe(); restored = Scene.restore(recipe); same = float(np.max(np.abs(scene.obs() - restored.obs())))
    act = teacher_action(scene, rng, 5); scene.step(act); restored.step(act); after = float(np.max(np.abs(scene.obs() - restored.obs())))
    replay = {'same_state_max_abs': same, 'next_transition_max_abs': after, 'passed': same < 1e-7 and after < 1e-7, 'method': 'same seed plus complete action-prefix replay, not pose-only restoration'}
    scene.close(); restored.close(); budget = Budget(100); budget.charge(99)
    try: budget.charge(2); blocked = False
    except RuntimeError: blocked = True
    check, goal = goal_task(941000, 0., 64, 14); initial_coverage = float(check.env._get_coverage()); trng = np.random.default_rng(941000 + 9876)
    for t in range(14): check.step(teacher_action(check, trng, t))
    reached = float(check.env._get_coverage()); check.close()
    coast = {}
    for d in [0., .25, .85]:       # what the damping parameter does: distance the block keeps travelling after the pusher stops
        s = Scene(5, d)
        for t in range(4): p = s.obs()[:2] * 512; b = s.obs()[2:4] * 512; s.step(p + (b - p) / np.linalg.norm(b - p) * 40)
        hold = s.obs()[:2] * 512; b0 = s.obs()[2:4] * 512
        for t in range(10): s.step(hold)
        coast[str(d)] = float(np.linalg.norm(s.obs()[2:4] * 512 - b0)); s.close()
    result = {'snapshot_replay': replay, 'budget_overrun_blocked': blocked, 'source': 'actual local reference checks', 'versions': versions(),
              'teacher_reachability': {'initial_coverage': initial_coverage, 'replayed_teacher_coverage': reached, 'success': reached > .95, 'scope': 'data-collection teacher accesses true state; not a deployed controller result'},
              'coast_px_10_steps_after_release': coast}
    if a.data:
        d = load_data(a.data); ids = {s: set(d['episode_id'][d['split'] == s].tolist()) for s in ['train', 'val', 'test']}
        result['split_disjoint'] = not any(ids[x] & ids[y] for x, y in [('train', 'val'), ('train', 'test'), ('val', 'test')]); result['episodes'] = int(len(d['split']))
    if a.model:
        agent = Agent(a.model); scene, goal = goal_task(99123, 0., agent.image_size); cur = agent.observe(scene); prev = cur.copy(); E = agent.E; K = 8; H = 4; T = 4; R = 2
        b = Budget(E * K * (H + T) * R); plan = agent.plan(prev, cur, goal, b, np.random.default_rng(15), cur[:2], H, K, R, None, T, .3); result['budget_charge_exact'] = b.used == E * K * (H + T) * R
        recipe = scene.recipe()      # fixed candidate set for the offline intervention audit; true outcomes never return to the controller
        cand = np.stack([plan['actions'], np.repeat((cur[:2] * 512)[None], H, 0), np.clip(plan['actions'] + np.array([12., 0.]), 16, 496)])
        mean, _, _ = agent.rollout(prev, cur, cand / 512, cur[:2]); model_cost = [float(c) for c in pose_cost(mean[:, -1], goal)]; true_cost = []
        for seq in cand:
            branch = Scene.restore(recipe)
            for act in seq: branch.step(act)
            true_cost.append(float(pose_cost(branch.obs(), goal))); branch.close()
        chosen = int(np.argmin(model_cost))
        result['counterfactual_candidates'] = {'count': 3, 'model_cost': model_cost, 'true_cost': true_cost, 'selected': chosen, 'decision_regret': true_cost[chosen] - min(true_cost),
                                               'diagnostic_model_calls': E * H * len(cand), 'scope': 'one fixed candidate set; diagnostic calls separate from controller budget, no performance claim'}
        if agent.mode == 'pixels':      # the deployed observer must run with privileged observation access disabled
            def forbidden(): raise RuntimeError('privileged state disabled')
            scene.obs = forbidden; result['pixel_observer_without_state_access'] = bool(np.isfinite(agent.observe(scene)).all())
        scene.close()
        if a.data:
            d = load_data(a.data, agent.mode == 'pixels'); mask = d['split'] == 'test'; truth = d['obs'][mask]; inputs = truth
            if agent.mode == 'pixels':
                inputs = estimate(agent.vision, d['images'][mask], 'cpu')
                result['heldout_visual_pose'] = {'position_rmse_px': float(np.sqrt(np.mean((inputs[..., :4] - truth[..., :4]) ** 2)) * 512),
                                                 'angle_mae_deg': float(np.degrees(np.arccos(np.clip(np.sum(inputs[..., 4:6] * truth[..., 4:6], -1), -1, 1))).mean())}
            rows = rows_of(inputs, d['actions'][mask], d['contact'][mask], truth)
            with torch.no_grad(): outs = [net(*rows[:4]) for net in agent.models]
            mean = np.mean([o[0].numpy() for o in outs], 0); prob = np.mean([o[1].sigmoid().numpy() for o in outs], 0); tgt = rows[4].numpy(); lab = rows[5].numpy()
            result['heldout_prediction'] = {'test_episodes': int(mask.sum()), 'transitions': len(tgt), 'position_rmse_px': float(np.sqrt(np.mean((mean[:, :4] - tgt[:, :4]) ** 2)) * 512),
                                            'angle_mae_deg': float(np.degrees(np.arccos(np.clip(np.sum(mean[:, 4:6] * tgt[:, 4:6], -1), -1, 1))).mean()),
                                            'near_contact_brier': float(np.mean((prob - lab) ** 2)), 'base_rate_brier': float(lab.mean() * (1 - lab.mean())), 'near_contact_auroc': auroc(prob, lab),
                                            'near_contact_rate': float(lab.mean()), 'source': 'untouched dataset test episodes; one-step diagnostic only'}
    save_json(out / 'audit.json', result); print(json.dumps(result, ensure_ascii=False), flush=True)
    if not replay['passed'] or not blocked or not result.get('split_disjoint', True) or not result.get('budget_charge_exact', True): raise RuntimeError('audit failed')


def mcnemar(x, y):
    n10 = int(((x == 1) & (y == 0)).sum()); n01 = int(((x == 0) & (y == 1)).sum()); n = n10 + n01
    return n10, n01, (min(1., 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.)


def report(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(a.input); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(root / 'episodes.csv'))); methods = list(dict.fromkeys(r['method'] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, key, label in zip(axes, ['success', 'max_coverage', 'model_calls'], ['Task success fraction', 'Mean best coverage in episode', 'Mean transition-model calls']):
        vals = [np.mean([float(r[key]) for r in rows if r['method'] == m]) for m in methods]
        bars = ax.bar(methods, vals); ax.set_ylabel(label); ax.tick_params(axis='x', labelrotation=45)
        if key != 'model_calls': ax.set_ylim(0, 1.08)
        for bar, m, v in zip(bars, methods, vals):
            text = f"{sum(int(r['success']) for r in rows if r['method'] == m)}/{sum(r['method'] == m for r in rows)}" if key == 'success' else f'{v:.2f}' if key == 'max_coverage' else f'{v / 1000:.1f}k'
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01 + (.01 if key != 'model_calls' else 0), text, ha='center', fontsize=7)
    fig.suptitle(f'Reference pilot: simulated PushT ({root.name})'); fig.tight_layout(); fig.savefig(out / 'pilot_comparison.png', dpi=160); plt.close(fig)
    tests = []
    for x, y in PAIRS:      # pairs fixed before looking at results; exact two-sided McNemar on per-task success
        if x in methods and y in methods:
            vx = np.array([int(r['success']) for r in rows if r['method'] == x]); vy = np.array([int(r['success']) for r in rows if r['method'] == y]); n10, n01, p = mcnemar(vx, vy)
            tests.append(dict(first=x, second=y, first_successes=int(vx.sum()), second_successes=int(vy.sum()), only_first=n10, only_second=n01, exact_mcnemar_p=p))
    save_json(out / 'paired_tests.json', {'tests': tests, 'note': 'Descriptive, single training seed, no multiplicity correction.'})
    save_json(out / 'report_manifest.json', {'source': str(root), 'rows': len(rows), 'figure': 'pilot_comparison.png', 'note': 'Descriptive pilot. No confidence intervals from one training seed.'})
    print(f'report rows={len(rows)}, figure={out / "pilot_comparison.png"}')


def gate(a):
    """Explicit, pre-declared criteria for enlarging the experiment. Reads summaries; edits nothing else."""
    root = Path(a.results); load = lambda name: json.loads((root / f'{a.prefix}{name}' / 'summary.json').read_text())['methods']
    nominal = load('state_d0.0_unlimited'); full = json.loads((root / f'{a.prefix}state_d0.0_unlimited' / 'summary.json').read_text())
    pixels = load('pixels_d0.0_unlimited'); held = json.loads(Path(a.audit).read_text())['heldout_prediction']; verification = json.loads(Path(a.verification).read_text())
    rate = lambda m: m['fixed1']['successes'] / m['fixed1']['tasks']
    checks = {'budget_and_observation_checks_passed': bool(verification['all_passed']), 'nominal_state_baseline_usable': rate(nominal) >= .25,
              'schedules_behave_differently': all(len(g) == 1 for g in full['identical_behaviour_groups']), 'near_contact_head_informative': held['near_contact_brier'] < held['base_rate_brier'] and held['near_contact_auroc'] >= .8,
              'nonperiodic_contact_trigger_observed': nominal['contact']['nonperiodic_trigger_rate'] > 0, 'visual_path_usable': rate(pixels) >= .10}
    shift = {}
    for d in a.shift_dampings:
        m = load(f'state_d{d}_unlimited'); shift[str(d)] = {'fixed1': f"{m['fixed1']['successes']}/{m['fixed1']['tasks']}", 'ended_worse_than_start': m['fixed1']['ended_worse_than_start'], 'baseline_usable': rate(m) >= .25}
    nominal_ok = all(checks.values()); eligible = ([0.0] if nominal_ok else []) + [float(d) for d, v in shift.items() if nominal_ok and v['baseline_usable']]
    result = {'kind': 'reference pilot readiness check', 'criteria': 'fixed1 success rate >= 0.25 for a usable state baseline and >= 0.10 for the pixel path; near-contact head must beat the base-rate Brier score with AUROC >= 0.8',
              'checks': checks, 'nominal': {'state_fixed1': f"{nominal['fixed1']['successes']}/{nominal['fixed1']['tasks']}", 'pixels_fixed1': f"{pixels['fixed1']['successes']}/{pixels['fixed1']['tasks']}",
                                            'contact_trigger_rate': nominal['contact']['nonperiodic_trigger_rate']}, 'shift_conditions': shift,
              'eligible_conditions': eligible, 'eligible_for_full': nominal_ok and all(v['baseline_usable'] for v in shift.values()),
              'override_note': 'run_protocol.py full --execute runs only dampings listed in eligible_conditions. An explicit --override-gate records deliberate exploratory execution.'}
    save_json(a.out, result); print(json.dumps(result, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=TITLE); sub = p.add_subparsers(dest='command', required=True); dev = dict(default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    c = sub.add_parser('collect'); c.add_argument('--out', required=True); c.add_argument('--episodes', type=int, default=2400); c.add_argument('--steps', type=int, default=32); c.add_argument('--image-size', type=int, default=64)
    c.add_argument('--seed', type=int, default=19); c.add_argument('--images', action='store_true'); c.add_argument('--dampings', type=float, nargs='+', default=NOMINAL_DAMPING)
    cp = sub.add_parser('collect-policy'); cp.add_argument('--model', required=True); cp.add_argument('--out', required=True); cp.add_argument('--episodes', type=int, default=1200); cp.add_argument('--steps', type=int, default=32)
    cp.add_argument('--seed', type=int, default=1); cp.add_argument('--eps', type=float, default=.3); cp.add_argument('--sigma', type=float, default=10.); cp.add_argument('--images', action='store_true')
    cp.add_argument('--dampings', type=float, nargs='+', default=NOMINAL_DAMPING); cp.add_argument('--tail', type=int, default=6); cp.add_argument('--shape', type=float, default=.3); cp.add_argument('--workers', type=int, default=1)
    t = sub.add_parser('train'); t.add_argument('--data', nargs='+', required=True); t.add_argument('--out', required=True); t.add_argument('--mode', choices=['state', 'pixels'], default='state'); t.add_argument('--epochs', type=int, default=60)
    t.add_argument('--vision-epochs', type=int, default=12); t.add_argument('--members', type=int, default=3); t.add_argument('--batch', type=int, default=512); t.add_argument('--hidden', type=int, default=256); t.add_argument('--seed', type=int, default=23)
    t.add_argument('--image-size', type=int, default=64); t.add_argument('--dampings', type=float, nargs='+', default=None); t.add_argument('--device', help='device for the image front-end; dynamics always trains on CPU', **dev)
    k = sub.add_parser('calibrate'); k.add_argument('--model', required=True); k.add_argument('--tasks', type=int, default=48); k.add_argument('--seed', type=int, default=52); k.add_argument('--rate', type=float, default=.25)
    k.add_argument('--max-commit', type=int, default=6); k.add_argument('--dampings', type=float, nargs='+', default=NOMINAL_DAMPING); k.add_argument('--tail', type=int, default=6); k.add_argument('--shape', type=float, default=.3)
    for name in ['evaluate', 'oracle']:
        e = sub.add_parser(name); e.add_argument('--out', required=True); e.add_argument('--tasks', type=int, default=8); e.add_argument('--steps', type=int, default=35); e.add_argument('--goal-steps', type=int, default=14)
        e.add_argument('--budget', type=int, default=0, help='hard cap on member-candidate-step predictions per task; 0 means unlimited'); e.add_argument('--horizon', type=int, default=6); e.add_argument('--tail', type=int, default=6)
        e.add_argument('--candidates', type=int, default=32); e.add_argument('--cem-iters', type=int, default=2); e.add_argument('--shape', type=float, default=.3); e.add_argument('--damping', type=float, default=0.); e.add_argument('--seed', type=int, default=71)
        e.add_argument('--max-commit', type=int, default=6); e.add_argument('--exhaust', choices=['suffix', 'hold'], default='suffix'); e.add_argument('--workers', type=int, default=1)
        if name == 'evaluate': e.add_argument('--model', required=True); e.add_argument('--methods', default=','.join(METHODS)); e.add_argument('--gif', type=int, default=0)
        else: e.add_argument('--methods', default='fixed1')
    u = sub.add_parser('audit'); u.add_argument('--out', required=True); u.add_argument('--data', nargs='+'); u.add_argument('--model'); u.add_argument('--seed', type=int, default=47)
    r = sub.add_parser('report'); r.add_argument('--input', required=True); r.add_argument('--out', required=True)
    g = sub.add_parser('gate'); g.add_argument('--results', default='results'); g.add_argument('--prefix', default=''); g.add_argument('--audit', default='audits/state_pilot/audit.json')
    g.add_argument('--verification', default='audits/final_verification.json'); g.add_argument('--shift-dampings', type=float, nargs='+', default=[.25, .85]); g.add_argument('--out', default='gate_report.json')
    a = p.parse_args()
    if a.command == 'collect' and (a.episodes < 10 or a.steps < 2): p.error('collect requires episodes>=10 and steps>=2')
    if a.command == 'train' and min(a.epochs, a.vision_epochs, a.members, a.batch) < 1: p.error('positive training values required')
    if a.command in ('evaluate', 'oracle') and min(a.tasks, a.steps, a.goal_steps, a.horizon, a.candidates) < 1: p.error('positive evaluation values required')
    globals()[a.command.replace('-', '_')](a)


if __name__ == '__main__': main()
