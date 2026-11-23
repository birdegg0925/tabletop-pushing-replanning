#!/usr/bin/env python3
"""Summarize held-out transition errors and Stage 2 near-contact triggers by stratum."""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REF = HERE.parent
PREFIX = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else ''      # Use a prefix to write results alongside the reference files.
RES = HERE / f'{PREFIX}results'; ROOT = HERE / f'{PREFIX}root'      # needs the on-policy data of every seed under ROOT/data, which rev_driver.py produces
sys.path.insert(0, str(REF))
import torch, run as R

STRATA = ('none', 'sustained', 'event')


def model_and_data(seed):
    if seed == 23: return REF / 'models' / f'{PREFIX}state_pilot' / 'model.pt', [REF / 'data' / f'{PREFIX}{n}' / 'dataset.npz' for n in ('explore', 'policy_r1', 'policy_r2', 'policy_r3')]
    return ROOT / 'models' / f'state_seed{seed}' / 'model.pt', [REF / 'data' / f'{PREFIX}explore' / 'dataset.npz'] + [ROOT / 'data' / f'policy_r{r}_seed{seed}' / 'dataset.npz' for r in (1, 2, 3)]


def error_strata(seed):
    model, data = model_and_data(seed); agent = R.Agent(model); d = R.load_data(data); m = d['split'] == 'test'; S, A, C = d['obs'][m], d['actions'][m], d['contact'][m]
    rows = R.rows_of(S, A, C, S)
    with torch.no_grad(): pred = np.mean([net(*rows[:4])[0].numpy() for net in agent.models], 0)
    tgt = rows[4].numpy(); cur = rows[1].numpy(); err = np.linalg.norm(pred[:, 2:4] - tgt[:, 2:4], axis=1) * 512; move = np.linalg.norm(tgt[:, 2:4] - cur[:, 2:4], axis=1) * 512
    before = (R.t_sdf(S[:, :-1]) <= R.NEAR_PX).reshape(-1); after = (R.t_sdf(S[:, 1:]) <= R.NEAR_PX).reshape(-1)
    lab = np.where(before != after, 2, np.where(before & after, 1, 0)); out = {}
    for i, name in enumerate(STRATA):
        s = lab == i; mv = s & (move > .5)
        out[name] = dict(n=int(s.sum()), share=float(s.mean()), err=float(err[s].mean()), move=float(move[s].mean()), moving_share=float((move[s] > .5).mean()),
                         err_moving=float(err[mv].mean()) if mv.any() else None, rel=float(err[mv].sum() / move[mv].sum()) if mv.any() else None)
    still = move <= .5; out['still'] = dict(n=int(still.sum()), err=float(err[still].mean())); out['moving'] = dict(n=int((~still).sum()), err=float(err[~still].mean()), rel=float(err[~still].sum() / move[~still].sum()))
    return out


def firing_strata(seed, arm):
    """Classify each eligible step of an arm by the true near-contact label around it; a replan less than six steps after the previous one is a firing."""
    fire = {k: 0 for k in STRATA}; quiet = {k: 0 for k in STRATA}; byt = np.zeros(35); nt = np.zeros(35)
    for r in csv.DictReader(open(RES / f'state_s{seed}' / 'episodes.csv')):
        if r['arm'] != arm or float(r['damping']) != 0: continue
        L = int(r['steps_stop']); rep = [int(c) for c in r['replan_bits'][:L]]; near = [int(c) for c in r['near_bits'][:L]]; last = 0
        for t in range(L): byt[t] += rep[t]; nt[t] += 1
        for t in range(1, L):
            if rep[t] and t - last >= 6: last = t; continue          # forced by the commitment limit
            b = near[t - 1]; a = near[t]; n = near[t + 1] if t + 1 < L else a
            k = 'event' if (b != a or a != n) else ('sustained' if a else 'none'); (fire if rep[t] else quiet)[k] += 1
            if rep[t]: last = t
    return dict(fire=fire, quiet=quiet, replan_by_step=(byt / np.maximum(nt, 1)).round(3).tolist(), alive_by_step=nt.tolist())


def success_by_step(seed, arms):
    out = {a: np.zeros(36) for a in arms}; n = {a: 0 for a in arms}
    for r in csv.DictReader(open(RES / f'state_s{seed}' / 'episodes.csv')):
        if r['arm'] in out and float(r['damping']) == 0 and r['draw'] in ('0', '1'):
            n[r['arm']] += 1
            if int(r['success']): out[r['arm']][int(r['steps_stop']):] += 1
    return {a: (out[a] / max(1, n[a])).round(4).tolist() for a in arms}


if __name__ == '__main__':
    seeds = sorted(int(p.name.split('_s')[1]) for p in RES.glob('state_s*') if (p / 'meta.json').exists()); res = dict(seeds=seeds, error={}, firing={}, cum={})
    arms = ['fixed1', 'fixed2', 'fixed3', 'fixed4', 'fixed6', 'contact_q75', 'random~contact_q75', 'error_q75']
    for s in seeds:
        res['error'][s] = error_strata(s); res['firing'][s] = {a: firing_strata(s, a) for a in ('contact_q75', 'random~contact_q75', 'error_q75')}; res['cum'][s] = success_by_step(s, arms)
    agg = {}
    for name in STRATA + ('still', 'moving'):
        agg[name] = {k: float(np.mean([res['error'][s][name][k] for s in seeds])) for k in res['error'][seeds[0]][name] if res['error'][seeds[0]][name][k] is not None}
    res['error_mean'] = agg; res['firing_total'] = {}
    for a in ('contact_q75', 'random~contact_q75', 'error_q75'):
        f = {k: sum(res['firing'][s][a]['fire'][k] for s in seeds) for k in STRATA}; q = {k: sum(res['firing'][s][a]['quiet'][k] for s in seeds) for k in STRATA}
        res['firing_total'][a] = dict(fire=f, quiet=q, fire_share={k: f[k] / max(1, sum(f.values())) for k in STRATA}, step_share={k: (f[k] + q[k]) / max(1, sum(f.values()) + sum(q.values())) for k in STRATA},
                                      rate_in={k: f[k] / max(1, f[k] + q[k]) for k in STRATA})
    (HERE / f'{PREFIX}strata.json').write_text(json.dumps(res, indent=1)); print(json.dumps(dict(error_mean=agg, firing_total=res['firing_total']), indent=1))
