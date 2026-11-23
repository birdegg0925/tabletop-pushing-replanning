#!/usr/bin/env python3
"""Stage 2 driver: trains the additional seeds with the unchanged reference CLI, then evaluates every model with rev_eval.py.
Resumable: a step whose output exists is skipped. See analysis_plan.md."""
import json, queue, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent; REF = HERE.parent
PREFIX = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else ''      # Use a prefix to write results alongside the reference files.
ROOT = HERE / f'{PREFIX}root'; RES = HERE / f'{PREFIX}results'; LOG = HERE / f'{PREFIX}logs'
DEVICE = sys.argv[sys.argv.index('--device') + 1] if '--device' in sys.argv else 'auto'      # image front-end only; the reference run used mps
WORKERS = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 16
STATE_SEEDS = [23, 29, 37, 43, 53, 59, 61, 67, 73, 79]; PIXEL_SEEDS = [23, 29, 37, 43, 53]
PY = sys.executable; RUN = str(REF / 'run.py'); EVAL = str(HERE / 'rev_eval.py'); training_done = threading.Event(); todo = queue.Queue(); lock = threading.Lock()


def sh(name, cmd):
    LOG.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with open(LOG / f'{name}.log', 'w') as f: r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    with lock: print(f'{time.strftime("%H:%M:%S")} {name} rc={r.returncode} {time.time() - t0:.0f}s', flush=True)
    if r.returncode: raise RuntimeError(name)


def model_path(mode, s): return (REF / 'models' / f'{PREFIX}{mode}_pilot' / 'model.pt') if s == 23 else (ROOT / 'models' / f'{mode}_seed{s}' / 'model.pt')


def train_seed(s):
    D = lambda n: ROOT / 'data' / n / 'dataset.npz'; M = lambda n: ROOT / 'models' / n / 'model.pt'; data = [REF / 'data' / f'{PREFIX}explore' / 'dataset.npz']
    if not M(f'state_r0_seed{s}').exists(): sh(f'train_state_r0_seed{s}', [PY, RUN, 'train', '--data', *data, '--out', M(f'state_r0_seed{s}').parent, '--mode', 'state', '--seed', s])
    for r in (1, 2, 3):
        prev = M(f'state_r{r - 1}_seed{s}'); d = D(f'policy_r{r}_seed{s}'); name = f'state_seed{s}' if r == 3 else f'state_r{r}_seed{s}'
        if not d.exists(): sh(f'collect_r{r}_seed{s}', [PY, RUN, 'collect-policy', '--model', prev, '--out', d.parent, '--episodes', 1200, '--seed', s * 10 + r, '--images', '--workers', 4])
        data.append(d)
        if not M(name).exists(): sh(f'train_{name}', [PY, RUN, 'train', '--data', *data, '--out', M(name).parent, '--mode', 'state', '--seed', s])
    todo.put(('state', s))
    if s in PIXEL_SEEDS:
        if not M(f'pixels_seed{s}').exists(): sh(f'train_pixels_seed{s}', [PY, RUN, 'train', '--data', *data, '--out', M(f'pixels_seed{s}').parent, '--mode', 'pixels', '--seed', s, '--vision-epochs', 30, '--device', DEVICE])
        todo.put(('pixels', s))


def evaluator():
    while True:
        item = todo.get()
        if item is None: return
        mode, s = item; out = RES / f'{mode}_s{s}'
        if (out / 'meta.json').exists(): continue
        w = WORKERS if training_done.is_set() else max(1, min(9, WORKERS))
        try:
            if mode == 'oracle': sh('eval_oracle', [PY, EVAL, '--oracle', '--out', out, '--workers', w])
            else: sh(f'eval_{mode}_s{s}', [PY, EVAL, '--model', model_path(mode, s), '--out', out, '--workers', w] + (['--selfcheck'] if s == 23 else []))
        except RuntimeError as e:
            with lock: print(f'FAILED {e}', flush=True)


if __name__ == '__main__':
    t0 = time.time(); th = threading.Thread(target=evaluator); th.start()
    todo.put(('state', 23)); todo.put(('pixels', 23)); todo.put(('oracle', 0))
    with ThreadPoolExecutor(3) as ex: futs = [ex.submit(train_seed, s) for s in STATE_SEEDS if s != 23]; errs = [f.exception() for f in futs]
    training_done.set(); todo.put(None); th.join()
    print(json.dumps(dict(done=True, hours=(time.time() - t0) / 3600, training_errors=[str(e) for e in errs if e])), flush=True)
