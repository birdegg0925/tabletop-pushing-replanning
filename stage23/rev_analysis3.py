#!/usr/bin/env python3
"""Summarize the Stage 2 and Stage 3 episode files using analysis_plan_stage3.md."""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else ''      # Use a prefix to write results alongside the reference files.
RES = HERE / f'{PREFIX}results'; R3 = HERE / f'{PREFIX}results3'
B = 10000; RNG = np.random.default_rng(20261118)
SCORES = ('error', 'disagreement', 'contact'); QS = (25, 50, 75, 90); RULES = [f'{s}_q{q}' for s in SCORES for q in QS] + ['geom']; PRIMARY = ['error_q75', 'disagreement_q75', 'contact_q75', 'geom']
MILD = ['1e-09', '1e-05', '0.001', '0.005']; KEYS = ('success', 'cov', 'final', 'rs', 'ss', 'r35', 'e10', 'n10', 'u10')


def cells(row):
    """Per-episode quantities. e10/n10: replans and steps in steps 0 to 9 while the episode is still running; u10: replans in steps 0 to 9 regardless."""
    n = int(row['steps_stop']); bits = row['replan_bits']; k = min(10, n)
    return [float(row['success']), float(row['best_cov35']), float(row['final_cov35']), float(row['replans_stop']), float(n), float(row['replans35']), bits[:k].count('1'), k, bits[:10].count('1')]


def load(mode):
    """dict[arm] -> dict[key] -> array (seeds, tasks); control arms are averaged over their timing draws. Only seeds with Stage 3 output are used."""
    dirs = [d for d in sorted(RES.glob(f'{mode}_s*'), key=lambda p: int(p.name.split('_s')[1])) if (R3 / d.name / 'shuffle.json').exists()]; seeds = [int(d.name.split('_s')[1]) for d in dirs]; out = {}; prof = {}
    for si, d in enumerate(dirs):
        acc = {}
        for path in (d / 'episodes.csv', R3 / d.name / 'shuffle.csv'):
            for r in csv.DictReader(open(path)):
                if float(r['damping']) != 0: continue
                a = r['arm']
                if a.split('~')[-1] not in RULES and not a.startswith('fixed'): continue
                acc.setdefault(a, {}).setdefault(int(r['task']), []).append(cells(r))
                if a.split('~')[-1] in RULES:                          # replans at each step over all episodes (full 35 steps), for the construction check
                    p = prof.setdefault(a, np.zeros((len(dirs), 35))); p[si] += np.array([int(c) for c in r['replan_bits']]) / (20 if '~' in a else 1)
        for a, by in acc.items():
            T = max(by) + 1; m = np.array([np.mean(by[t], 0) for t in range(T)]); o = out.setdefault(a, {k: np.full((len(dirs), T), np.nan) for k in KEYS})
            for j, k in enumerate(KEYS): o[k][si] = m[:, j]
    return out, seeds, prof


def ci(d, idx, scale):
    si, ti = idx; stat = np.array([d[np.ix_(si[b], ti[b])].mean() for b in range(B)]) * scale
    return dict(diff=float(d.mean() * scale), lo=float(np.quantile(stat, .025)), hi=float(np.quantile(stat, .975)), p=float(min(1., max(1 / B, 2 * min(np.mean(stat <= 0), np.mean(stat >= 0))))))


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.
    for rank, i in enumerate(order): run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1., run)
    return adj.tolist()


def analyse(mode):
    D, seeds, prof = load(mode)
    if not seeds: return None
    S, T = D['fixed1']['success'].shape; idx = (RNG.integers(0, S, (B, S)), RNG.integers(0, T, (B, T))); out = dict(seeds=seeds, n_seeds=S, tasks=T, arms={}, sweep={})
    rate = lambda a: float(D[a]['rs'].sum() / D[a]['ss'].sum()); early = lambda a: float(D[a]['e10'].sum() / D[a]['n10'].sum())
    for r in RULES:
        sh, rn = 'shuffle~' + r, 'random~' + r; e = {}
        for name, a in (('rule', r), ('shuffle', sh), ('random', rn)):
            ps = D[a]['success'].sum(1)
            e[name] = dict(mean=float(ps.mean()), min=float(ps.min()), max=float(ps.max()), per_seed=[round(float(v), 2) for v in ps], cov=float(D[a]['cov'].mean()), rate=rate(a), early=early(a),
                           rate35=float(D[a]['r35'].mean() / 35), u10=float(D[a]['u10'].mean() / 10))
        # construction check: the control uses the rule's schedules, so replans at every step of the full episode are identical
        e['profile_max_abs_diff'] = float(np.abs(prof[r] - prof[sh]).max()); e['profile_random_max_abs_diff'] = float(np.abs(prof[r] - prof[rn]).max() / T)
        for key, scale, tag in (('success', T, ''), ('cov', 1., '_cov')):
            e['rule_shuffle' + tag] = ci(D[r][key] - D[sh][key], idx, scale); e['shuffle_random' + tag] = ci(D[sh][key] - D[rn][key], idx, scale); e['rule_random' + tag] = ci(D[r][key] - D[rn][key], idx, scale)
        e['profile'] = dict(rule=(prof[r].mean(0) / T).tolist(), shuffle=(prof[sh].mean(0) / T).tolist(), random=(prof[rn].mean(0) / T).tolist())   # share of all episodes replanning at each step
        e['seeds_rule_behind'] = int((D[r]['success'].sum(1) < D[sh]['success'].sum(1)).sum())
        e['seeds_rule_ahead'] = int((D[r]['success'].sum(1) > D[sh]['success'].sum(1)).sum()); e['seeds_rule_ahead_cov'] = int((D[r]['cov'].mean(1) > D[sh]['cov'].mean(1)).sum())
        e['seeds_rule_ahead_random'] = int((D[r]['success'].sum(1) > D[rn]['success'].sum(1)).sum())
        out['sweep'][r] = e
    fam = [(r, t) for t in ('', '_cov') for r in PRIMARY]                 # primary family: four rules x two outcomes, Holm across the eight
    for (r, t), a in zip(fam, holm([out['sweep'][r]['rule_shuffle' + t]['p'] for r, t in fam])): out['sweep'][r]['rule_shuffle' + t]['p_holm8'] = a
    for t in ('', '_cov'):                                                 # reference only: Holm across the 13 arms
        for k in ('rule_shuffle', 'shuffle_random'):
            for r, a in zip(RULES, holm([out['sweep'][r][k + t]['p'] for r in RULES])): out['sweep'][r][k + t]['p_holm13'] = a
    # Final-step coverage was added during analysis as an exploratory outcome.
    out['final_cov'] = dict(fixed1=float(D['fixed1']['final'].mean()), fixed4=float(D['fixed4']['final'].mean()), F=ci(D['fixed1']['final'] - D['fixed4']['final'], idx, 1.),
                            above95_fixed1=float((D['fixed1']['final'] > .95).sum(1).mean()), above95_fixed4=float((D['fixed4']['final'] > .95).sum(1).mean()),
                            F95=ci((D['fixed1']['final'] > .95).astype(float) - (D['fixed4']['final'] > .95).astype(float), idx, T))
    return out


def mild():
    dirs = [d for d in sorted(R3.glob('state_s*'), key=lambda p: int(p.name.split('_s')[1])) if (d / 'mild.csv').exists()]
    if not dirs or not (R3 / 'oracle_s0' / 'mild.csv').exists(): return None
    S = len(dirs); A = {}
    for si, d in enumerate(dirs):
        for r in csv.DictReader(open(d / 'mild.csv')):
            k = (r['arm'], f"{float(r['damping']):g}"); o = A.setdefault(k, {x: np.zeros((S, 64)) for x in ('success', 'cov', 'el', 'np', 'rs', 'ss')}); t = int(r['task']); o['rs'][si, t] = float(r['replans_stop']); o['ss'][si, t] = float(r['steps_stop'])
            o['success'][si, t] = float(r['success']); o['cov'][si, t] = float(r['best_cov35']); o['el'][si, t] = float(r['eligible_stop']); o['np'][si, t] = float(r['nonper_stop'])
    idx = (RNG.integers(0, S, (B, S)), RNG.integers(0, 64, (B, 64))); orc = {}
    for r in csv.DictReader(open(R3 / 'oracle_s0' / 'mild.csv')): k = f"{float(r['damping']):g}"; orc[k] = orc.get(k, 0) + int(r['success'])
    out = dict(seeds=[int(d.name.split('_s')[1]) for d in dirs], coast=json.loads((R3 / 'oracle_s0' / 'mild.json').read_text())['coast_px'], oracle=orc, values={})
    for d in MILD:
        v = {}
        for a in ('fixed1', 'fixed4', 'error_q75'):
            ps = A[(a, d)]['success'].sum(1); v[a] = dict(mean=float(ps.mean()), min=float(ps.min()), max=float(ps.max()), cov=float(A[(a, d)]['cov'].mean()), rate=float(A[(a, d)]['rs'].sum() / A[(a, d)]['ss'].sum()))
        v['trigger'] = float(A[('error_q75', d)]['np'].sum() / A[('error_q75', d)]['el'].sum()); v['F'] = ci(A[('fixed1', d)]['success'] - A[('fixed4', d)]['success'], idx, 64); out['values'][d] = v
    return out


def floors():
    out = {}
    for mode in ('state', 'pixels'):
        fs = [json.loads(p.read_text()) for p in sorted(R3.glob(f'{mode}_s*/floor.json'))]
        if not fs: continue
        m = lambda f: float(np.mean([f(x) for x in fs]))
        out[mode] = dict(n=len(fs), floor_px=m(lambda x: x['floor_median'] * 512), floor_p90_px=m(lambda x: x['floor_p90'] * 512), moving_px=m(lambda x: x['moving_median'] * 512), rest_share=m(lambda x: x['rest_share']),
                         thr_px={str(q): m(lambda x: x['thresholds'][str(q)] * 512) for q in QS}, multiple={str(q): m(lambda x: x['multiples'][str(q)]) for q in QS},
                         multiple_min={str(q): min(x['multiples'][str(q)] for x in fs) for q in QS}, multiple_max={str(q): max(x['multiples'][str(q)] for x in fs) for q in QS},
                         rest_above={str(q): m(lambda x: x['share_rest_above'][str(q)]) for q in QS})
    return out


if __name__ == '__main__':
    res = {m: analyse(m) for m in ('state', 'pixels')}; res['mild'] = mild(); res['floor'] = floors()
    (HERE / f'{PREFIX}stage3.json').write_text(json.dumps(res, indent=1))
    for mode in ('state', 'pixels'):
        r = res[mode]
        if not r: continue
        print(f"== {mode}: seeds {r['seeds']}")
        for a in RULES:
            e = r['sweep'][a]; f = lambda v, k=2: f"{v['diff']:+.{k}f} [{v['lo']:+.{k}f},{v['hi']:+.{k}f}]"
            print(f"  {a:18s} rule {e['rule']['mean']:5.1f} shuf {e['shuffle']['mean']:5.1f} rand {e['random']['mean']:5.1f} | early {e['rule']['early']:.2f}/{e['shuffle']['early']:.2f}/{e['random']['early']:.2f} profdiff {e['profile_max_abs_diff']:.1e}"
                  f" | r-s {f(e['rule_shuffle'])} p {e['rule_shuffle']['p']:.3f} | s-r {f(e['shuffle_random'])} | cov r-s {f(e['rule_shuffle_cov'], 4)} s-r {f(e['shuffle_random_cov'], 4)} | holm8 {e['rule_shuffle'].get('p_holm8')}/{e['rule_shuffle_cov'].get('p_holm8')}")
        print('  final coverage', r['final_cov'])
    print(json.dumps(res['mild'], indent=1) if res['mild'] else 'Mild-damping summary: required episode inputs are absent.'); print(json.dumps(res['floor'], indent=1))
