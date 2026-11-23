#!/usr/bin/env python3
"""Stage 3 runs (analysis_plan_stage3.md): schedule-shuffled control, milder damping values, floor of the execution-error score.
Reads Stage 2 episode files and writes Stage 3 results. Existing outputs are reused."""
import argparse, csv, json, os, sys, time
from pathlib import Path
import numpy as np
import rev_eval as E

R = E.R; HERE = Path(__file__).resolve().parent; RES = HERE / f'{E.PREFIX}results'; OUT = HERE / f'{E.PREFIX}results3'
RULES = [f'{s}_q{q}' for s in E.SCORES for q in E.QS] + ['geom']; MILD = (1e-9, 1e-5, 1e-3, 5e-3); DRAWS = 20
STATE_SEEDS = [23, 29, 37, 43, 53, 59, 61, 67, 73, 79]; PIXEL_SEEDS = [23, 29, 37, 43, 53]


def stage2(mode, seed):
    d = RES / f'{mode}_s{seed}'; meta = json.loads((d / 'meta.json').read_text()); rows = {}
    for r in csv.DictReader(open(d / 'episodes.csv')):
        if r['arm'] in RULES and float(r['damping']) == 0 and int(r['draw']) == 0: rows.setdefault(r['arm'], {})[int(r['task'])] = r
    return meta, rows


def derangement(rng, n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p


def write(path, rows, extra=()):
    with open(path, 'w', newline='') as f: w = csv.DictWriter(f, fieldnames=E.FIELDS + list(extra)); w.writeheader(); w.writerows(rows)


def shuffled(mode, seed, workers, tasks=E.TASKS):
    out = OUT / f'{mode}_s{seed}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'shuffle.json').exists(): return
    t0 = time.perf_counter(); meta, s2 = stage2(mode, seed); model = str(E.model_path(mode, seed))
    # 1. replaying a rule's own schedule on its own task must reproduce the rule's episode exactly
    chk = E.pool_run(model, [('ep', t, E.TEST_SEED, [(0., dict(name='own~' + a, method='schedule', bits=s2[a][t]['replan_bits']), [0], False) for a in RULES]) for t in (0, 1)], min(2, workers)); n = 0
    for r in chk:
        ref = s2[r['arm'][4:]][r['task']]
        for k in ('success', 'replans35', 'units35'): assert int(r[k]) == int(float(ref[k])), (r['arm'], r['task'], k, r[k], ref[k])
        assert r['replan_bits'] == ref['replan_bits'] and r['cov_series'] == ref['cov_series'], (r['arm'], r['task']); n += 1
    # 2. the control: each task runs the schedule the rule produced on another task
    perm = {(a, k): derangement(np.random.default_rng([seed, ai, k]), tasks) for ai, a in enumerate(RULES) for k in range(1, DRAWS + 1)}
    jobs = [('ep', t, E.TEST_SEED, [(0., dict(name='shuffle~' + a, method='schedule', bits=s2[a][int(perm[(a, k)][t])]['replan_bits']), [k], False) for a in RULES for k in range(1, DRAWS + 1)])
            for t in range(tasks)]
    rows = E.pool_run(model, jobs, workers); bad = 0
    for r in rows:
        a = r['arm'][8:]; r['donor'] = int(perm[(a, r['draw'])][r['task']]); bad += int(r['replan_bits'] != s2[a][r['donor']]['replan_bits'] or r['donor'] == r['task'])
    assert bad == 0, bad
    for a in RULES:                                                    # same schedules, each used once per draw
        for k in range(1, DRAWS + 1): assert sorted(r['donor'] for r in rows if r['arm'] == 'shuffle~' + a and r['draw'] == k) == list(range(tasks))
    write(out / 'shuffle.csv', rows, ['donor'])
    (out / 'shuffle.json').write_text(json.dumps(dict(model=os.path.relpath(model, HERE), own_schedule_reproduced=n, episodes=len(rows), draws=DRAWS, seconds=time.perf_counter() - t0), indent=1))
    print(json.dumps(dict(part='shuffle', mode=mode, seed=seed, episodes=len(rows), own_checked=n, seconds=round(time.perf_counter() - t0))), flush=True)


def mild(seed, workers, tasks=E.TASKS):
    out = OUT / f'state_s{seed}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'mild.csv').exists(): return
    t0 = time.perf_counter(); meta, _ = stage2('state', seed)
    arms = [dict(name='fixed1', method='fixed1'), dict(name='fixed4', method='fixed4'), dict(name='error_q75', method='error', cal={'error_threshold': meta['thresholds']['error']['75']})]
    rows = E.pool_run(str(E.model_path('state', seed)), [('ep', t, E.TEST_SEED, [(d, arm, [0], False) for d in MILD for arm in arms]) for t in range(tasks)], workers)
    write(out / 'mild.csv', rows); print(json.dumps(dict(part='mild', seed=seed, episodes=len(rows), seconds=round(time.perf_counter() - t0))), flush=True)


def mild_oracle(workers, tasks=E.TASKS):
    out = OUT / 'oracle_s0'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'mild.csv').exists(): return
    t0 = time.perf_counter(); rows = E.pool_run(None, [('ep', t, E.TEST_SEED, [(d, dict(name='fixed1', method='fixed1'), [0], True) for d in MILD]) for t in range(tasks)], workers); coast = {}
    for d in MILD:                                                     # same audit procedure as rev_eval.run_oracle
        s = R.Scene(5, d)
        for t in range(4): p = s.obs()[:2] * 512; b = s.obs()[2:4] * 512; s.step(p + (b - p) / np.linalg.norm(b - p) * 40)
        hold = s.obs()[:2] * 512; b0 = s.obs()[2:4] * 512
        for t in range(10): s.step(hold)
        coast[f'{d:g}'] = float(np.linalg.norm(s.obs()[2:4] * 512 - b0)); s.close()
    write(out / 'mild.csv', rows); (out / 'mild.json').write_text(json.dumps(dict(coast_px=coast, seconds=time.perf_counter() - t0), indent=1))
    print(json.dumps(dict(part='mild_oracle', episodes=len(rows), seconds=round(time.perf_counter() - t0))), flush=True)


def floor(mode, seed, workers):
    out = OUT / f'{mode}_s{seed}'; out.mkdir(parents=True, exist_ok=True)
    if (out / 'floor.json').exists(): return
    meta, _ = stage2(mode, seed); arm = dict(name='fixed6', method='fixed6', motion=True)
    res = E.pool_run(str(E.model_path(mode, seed)), [('scores', t, E.CAL_SEED, [(0., arm, [0], False)]) for t in range(E.CAL_TASKS)], workers); e_all = []; m_all = []
    for row, sc, mv in res:
        for (e, _, _), m in list(zip(sc, mv))[:row['steps_stop']]:   # as in calibration: steps up to first success
            if e is not None and m is not None: e_all.append(e); m_all.append(m)
    e = np.array(e_all); m = np.array(m_all); rest = m < .5; thr = meta['thresholds']['error']
    for q in E.QS: assert abs(float(np.quantile(e, q / 100)) - thr[str(q)]) < 1e-12, (q, float(np.quantile(e, q / 100)), thr[str(q)])    # same scores as the calibration run
    f = dict(n=int(len(e)), n_rest=int(rest.sum()), rest_share=float(rest.mean()), floor_median=float(np.median(e[rest])), floor_p90=float(np.quantile(e[rest], .9)),
             moving_median=float(np.median(e[~rest])), thresholds={str(q): thr[str(q)] for q in E.QS}, multiples={str(q): thr[str(q)] / float(np.median(e[rest])) for q in E.QS},
             share_rest_above={str(q): float((e[rest] > thr[str(q)]).mean()) for q in E.QS})
    (out / 'floor.json').write_text(json.dumps(f, indent=1)); print(json.dumps(dict(part='floor', mode=mode, seed=seed, floor_median=f['floor_median'], n_rest=f['n_rest'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=16); p.add_argument('--prefix', default=''); a = p.parse_args(); t0 = time.time()
    for s in STATE_SEEDS: shuffled('state', s, a.workers)
    mild_oracle(a.workers)
    for s in STATE_SEEDS: mild(s, a.workers)
    for s in STATE_SEEDS: floor('state', s, a.workers)
    for s in PIXEL_SEEDS: floor('pixels', s, a.workers)
    for s in PIXEL_SEEDS: shuffled('pixels', s, a.workers)
    print(json.dumps(dict(done=True, hours=(time.time() - t0) / 3600)), flush=True)
