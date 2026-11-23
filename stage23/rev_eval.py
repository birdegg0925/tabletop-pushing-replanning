#!/usr/bin/env python3
"""Run the comparisons defined in the Stage 2 analysis plan (analysis_plan.md).
Imports the reference implementation unchanged; adds an episode loop with full-length episodes, per-arm thresholds,
random-control probability and timing draws, and per-arm search size."""
import argparse, csv, json, os, platform, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REF = HERE.parent                         # this folder sits inside the reference implementation
PREFIX = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else ''      # Use a prefix to write results alongside the reference files.
sys.path.insert(0, str(REF))
import torch, run as R

TEST_SEED, CAL_SEED, TASKS, CAL_TASKS, STEPS, MAXC = 83, 52, 64, 48, 35, 6
QS = (25, 50, 75, 90); SCORES = ('error', 'disagreement', 'contact'); SHIFT = (0.02, 0.05, 0.10, 0.25, 0.85)
EQUAL = [(1, 8, 2), (1, 16, 2), (2, 16, 2), (2, 64, 2), (4, 64, 2), (4, 128, 2), (4, 64, 4), (4, 32, 8)]
FIELDS = ['task', 'seed', 'damping', 'arm', 'draw', 'success', 'steps_stop', 'units_stop', 'replans_stop', 'eligible_stop', 'nonper_stop', 'best_cov_stop', 'progress_stop',
          'units35', 'replans35', 'best_cov35', 'final_cov35', 'hold35', 'replan_bits', 'near_bits', 'cov_series']
_W = {}


def task_seed(base, task): return 900000 + base * 1000 + task


def model_path(mode, seed):
    """Seed 23 is the pilot model of the reference implementation; the other seeds are trained by rev_driver.py."""
    return (REF / 'models' / f'{PREFIX}{mode}_pilot' / 'model.pt') if seed == 23 else (HERE / f'{PREFIX}root' / 'models' / f'{mode}_seed{seed}' / 'model.pt')


def episode(agent, task, seed, damping, arm, draw=0, early_stop=False, scores=False):
    """Same control loop as run.run_episode (no cap, no exploration), continued past first success unless early_stop."""
    method = arm['method']; cal = arm.get('cal', {}); K = arm.get('K', 32); iters = arm.get('iters', 2)
    bits = arm.get('bits'); motion = arm.get('motion'); moved = None; mv = []   # Stage 3: replay a given replan schedule; log block displacement
    scene, goal = R.goal_task(seed, damping, agent.image_size, 14); budget = R.Budget(10 ** 12)
    rng = np.random.default_rng(seed); trng = np.random.default_rng(seed + 77) if draw == 0 else np.random.default_rng([seed, 77, draw])
    prev = agent.observe(scene); cur = prev.copy(); last = cur[:2].copy(); plan = None; age = 0; replans = 0; hold = 0; nonper = 0; eligible = 0
    truth = scene.obs(); initial = float(R.pose_cost(truth, goal)); best = float(scene.env._get_coverage()); stop = None; rep = []; near = []; covs = []; sc = []
    for t in range(STEPS):
        diag = R.trigger_diagnostics('fixed99' if bits else method, age, plan, cur, cal, MAXC, trng); did = 0
        if bits and plan is not None: assert age < MAXC or bits[t] == '1', (t, age); diag['trigger'] = bits[t] == '1'
        if motion: mv.append(moved)
        if plan is not None and method in R.ADAPTIVE and age < MAXC: eligible += 1; nonper += int(diag['trigger'])
        if diag['trigger']:
            warm = None
            if plan is not None and age < len(plan['actions']):
                rest = plan['actions'][age:] / 512; warm = np.concatenate([rest, np.repeat(rest[-1:], 6 - len(rest), 0)])
            plan = agent.plan(prev, cur, goal, budget, rng, last, 6, K, iters, warm, 6, .3); age = 0; replans += 1; did = 1
        if age >= len(plan['actions']): act = cur[:2] * 512; hold += 1
        else: act = plan['actions'][age]
        before = cur.copy(); scene.step(act); last = (np.clip(act, 16, 496) / 512).astype(np.float32); prev, cur = before, agent.observe(scene); age += 1
        old = truth; truth = scene.obs(); moved = float(np.linalg.norm(truth[2:4] - old[2:4]) * 512); cov = float(scene.env._get_coverage()); best = max(best, cov); rep.append(did); near.append(int(R.proximity(scene.env))); covs.append(cov)
        if scores: sc.append((diag['error'], diag['disagreement'], diag['contact']))
        if cov > .95 and stop is None:
            stop = dict(success=1, steps_stop=t + 1, units_stop=budget.used, replans_stop=replans, eligible_stop=eligible, nonper_stop=nonper, best_cov_stop=best,
                        progress_stop=1 - float(R.pose_cost(truth, goal)) / initial)
            if early_stop: break
    if stop is None:
        stop = dict(success=0, steps_stop=len(rep), units_stop=budget.used, replans_stop=replans, eligible_stop=eligible, nonper_stop=nonper, best_cov_stop=best,
                    progress_stop=1 - float(R.pose_cost(truth, goal)) / initial)
    scene.close()
    row = dict(task=task, seed=seed, damping=damping, arm=arm['name'], draw=draw, **stop, units35=budget.used, replans35=replans, best_cov35=best, final_cov35=cov, hold35=hold,
               replan_bits=''.join(map(str, rep)), near_bits=''.join(map(str, near)), cov_series=' '.join(f'{c:.3f}' for c in covs))
    if motion: return row, sc, mv
    return (row, sc) if scores else row


def solve_p(rate):
    """Probability p whose long-run replans per step, p / (1 - (1 - p)^6), equals `rate`."""
    if rate <= 1 / 6 + 1e-6: return 0.
    if rate >= 1: return 1.
    lo, hi = 1e-6, 1.
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid / (1 - (1 - mid) ** 6) < rate: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def _init(model):
    torch.set_num_threads(1); _W['agent'] = R.Agent(model) if model else R.OracleAgent()


def _job(job):
    kind, task, base, items = job; agent = _W['agent']; seed = task_seed(base, task); out = []
    for damping, arm, draws, early in items:
        for dr in draws:
            if kind == 'scores': out.append(episode(agent, task, seed, damping, arm, dr, early, True))
            else: out.append(episode(agent, task, seed, damping, arm, dr, early))
    return out


def pool_run(model, jobs, workers):
    if workers <= 1: _init(model); return [r for j in jobs for r in _job(j)]
    with Pool(workers, initializer=_init, initargs=(model,)) as pool: return [r for chunk in pool.map(_job, jobs, chunksize=1) for r in chunk]


def rate_of(rows, arm, d=0.): r = [x for x in rows if x['arm'] == arm and x['damping'] == d]; return sum(x['replans_stop'] for x in r) / sum(x['steps_stop'] for x in r)


def write_rows(path, rows):
    with open(path, 'w', newline='') as f: w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def selfcheck(model, workers):
    """The new loop must reproduce run.run_episode on the pilot schedules, and the pilot calibration on the pilot model."""
    agent = R.Agent(model); cal = agent.cal; cfg = argparse.Namespace(goal_steps=14, budget=0, steps=35, horizon=6, candidates=32, cem_iters=2, max_commit=6, exhaust='suffix', tail=6, shape=.3); n = 0
    for task in range(4):
        seed = task_seed(71, task)
        for m in R.METHODS:
            ref, _, _, _ = R.run_episode(agent, task, seed, 0., m, cfg, cal); new = episode(agent, task, seed, 0., dict(name=m, method=m, cal=cal))
            for a, b in [('success', 'success'), ('steps', 'steps_stop'), ('model_calls', 'units_stop'), ('replans', 'replans_stop'), ('max_coverage', 'best_cov_stop'), ('nonperiodic_triggers', 'nonper_stop'), ('eligible_steps', 'eligible_stop')]:
                assert abs(float(ref[a]) - float(new[b])) < 1e-9, (task, m, a, ref[a], new[b])
            n += 1
    return n


def calibrate(model, workers):
    arm = dict(name='fixed6', method='fixed6'); jobs = [('scores', t, CAL_SEED, [(0., arm, [0], False)]) for t in range(CAL_TASKS)]
    res = pool_run(model, jobs, workers); v = {s: [] for s in SCORES}
    for row, sc in res:
        for e, d, c in sc[:row['steps_stop']]:          # the pilot calibration stops at first success
            if e is not None: v['error'].append(e); v['disagreement'].append(d); v['contact'].append(c)
    return {s: {q: float(np.quantile(np.array(v[s]), q / 100)) for q in QS} for s in SCORES}, {s: len(v[s]) for s in SCORES}


def timing(model):
    torch.set_num_threads(1); agent = R.Agent(model); ms = []; vis = []
    for task in range(8):
        scene, goal = R.goal_task(task_seed(CAL_SEED, task), 0., agent.image_size, 14); cur = agent.observe(scene); rng = np.random.default_rng(task)
        for _ in range(5):
            t0 = time.perf_counter(); agent.plan(cur, cur, goal, R.Budget(10 ** 12), rng, cur[:2].copy(), 6, 32, 2, None, 6, .3); ms.append(1000 * (time.perf_counter() - t0))
            t0 = time.perf_counter(); agent.observe(scene); vis.append(1000 * (time.perf_counter() - t0))
        scene.close()
    return dict(plan_ms_mean=float(np.mean(ms)), plan_ms_p95=float(np.quantile(ms, .95)), observe_ms_mean=float(np.mean(vis)), mode=agent.mode, threads=1,
                machine=platform.machine(), processor=platform.processor(), system=platform.platform(), units_per_plan=agent.E * 32 * 12 * 2)


def run_model(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter(); meta = dict(model=os.path.relpath(a.model, HERE), model_sha256=R.sha(a.model), test_seed=TEST_SEED, cal_seed=CAL_SEED, tasks=a.tasks)
    if a.selfcheck: meta['selfcheck_episodes'] = selfcheck(a.model, a.workers)
    thr, n = calibrate(a.model, a.workers); meta['thresholds'] = thr; meta['calibration_scores'] = n
    pilot = R.Agent(a.model).cal
    if a.selfcheck and pilot:
        for s in SCORES: assert abs(thr[s][75] - pilot[s + '_threshold']) < 1e-12, (s, thr[s][75], pilot[s + '_threshold'])
        meta['calibration_matches_pilot'] = True
    arms = [dict(name=f'fixed{k}', method=f'fixed{k}') for k in (1, 2, 3, 4, 6)]; rules = []
    for s in SCORES:
        for q in QS: rules.append(dict(name=f'{s}_q{q}', method=s, cal={s + '_threshold': thr[s][q]}))
    rules.append(dict(name='geom', method='geom'))
    equal = [dict(name=f'fixed{p}_K{K}_i{i}', method=f'fixed{p}', K=K, iters=i) for p, K, i in EQUAL]
    err75 = next(r for r in rules if r['name'] == 'error_q75')
    items = [(0., arm, [0], False) for arm in arms + rules + equal] + [(d, arm, [0], False) for d in SHIFT for arm in (arms[0], arms[3], err75)]
    rows = pool_run(a.model, [('ep', t, TEST_SEED, items) for t in range(a.tasks)], a.workers); meta['seconds_block_a'] = time.perf_counter() - t0
    match = {}
    for r in rules:                                      # matched random control: tune on realised rate only, then 20 fresh draws
        target = rate_of(rows, r['name']); p0 = solve_p(target); match[r['name']] = dict(target=target, p0=p0)
    tune = [(0., dict(name='tune~' + r['name'], method='random', cal={'random_rate': match[r['name']]['p0']}), list(range(101, 105)), False) for r in rules]
    trows = pool_run(a.model, [('ep', t, TEST_SEED, tune) for t in range(a.tasks)], a.workers)
    for r in rules:
        m = match[r['name']]; m['tune_rate'] = rate_of(trows, 'tune~' + r['name']); m['p'] = solve_p(min(1., max(1 / 6, m['p0'] / (1 - (1 - m['p0']) ** 6) + m['target'] - m['tune_rate']))) if m['p0'] > 0 else 0.
    ctrl = [(0., dict(name='random~' + r['name'], method='random', cal={'random_rate': match[r['name']]['p']}), list(range(1, 21)), False) for r in rules]
    crow = pool_run(a.model, [('ep', t, TEST_SEED, ctrl) for t in range(a.tasks)], a.workers)
    for r in rules:
        m = match[r['name']]; m['control_rate'] = rate_of(crow, 'random~' + r['name']); m['matched'] = bool(abs(m['control_rate'] - m['target']) <= .015)
    meta['matching'] = match; meta['seconds'] = time.perf_counter() - t0; meta['timing'] = timing(a.model); meta['versions'] = R.versions()
    write_rows(out / 'episodes.csv', rows + crow); (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(dict(out=str(out), episodes=len(rows) + len(crow), seconds=round(meta['seconds']), unmatched=[k for k, v in match.items() if not v['matched']])), flush=True)


def run_oracle(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.perf_counter()
    items = [(0., dict(name=f'fixed{k}', method=f'fixed{k}'), [0], True) for k in (1, 2, 3, 4, 6)] + [(d, dict(name='fixed1', method='fixed1'), [0], True) for d in SHIFT]
    rows = pool_run(None, [('ep', t, TEST_SEED, items) for t in range(a.tasks)], a.workers); coast = {}
    for d in (0.,) + SHIFT:                               # distance the block keeps travelling after the pusher stops (audit procedure)
        s = R.Scene(5, d)
        for t in range(4): p = s.obs()[:2] * 512; b = s.obs()[2:4] * 512; s.step(p + (b - p) / np.linalg.norm(b - p) * 40)
        hold = s.obs()[:2] * 512; b0 = s.obs()[2:4] * 512
        for t in range(10): s.step(hold)
        coast[str(d)] = float(np.linalg.norm(s.obs()[2:4] * 512 - b0)); s.close()
    write_rows(out / 'episodes.csv', rows); (out / 'meta.json').write_text(json.dumps(dict(kind='simulator-scored planner', tasks=a.tasks, test_seed=TEST_SEED, coast_px=coast, seconds=time.perf_counter() - t0), indent=1))
    print(json.dumps(dict(out=str(out), episodes=len(rows), seconds=round(time.perf_counter() - t0))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--model'); p.add_argument('--out', required=True); p.add_argument('--workers', type=int, default=8); p.add_argument('--tasks', type=int, default=TASKS)
    p.add_argument('--oracle', action='store_true'); p.add_argument('--selfcheck', action='store_true'); p.add_argument('--prefix', default=''); a = p.parse_args()
    run_oracle(a) if a.oracle else run_model(a)
