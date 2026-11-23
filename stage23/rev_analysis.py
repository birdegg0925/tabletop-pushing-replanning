#!/usr/bin/env python3
"""Summarize Stage 2 episode results using the comparisons in analysis_plan.md."""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else ''      # Use a prefix to write results alongside the reference files.
RES = HERE / f'{PREFIX}results'
B = 10000; RNG = np.random.default_rng(20261118); UNITS = 2304
SCORES = ('error', 'disagreement', 'contact'); QS = (25, 50, 75, 90); RULES = [f'{s}_q{q}' for s in SCORES for q in QS] + ['geom']; FIXED = [f'fixed{k}' for k in (1, 2, 3, 4, 6)]
SHIFT = ['0.02', '0.05', '0.1', '0.25', '0.85']
EQ = {576: {1: 'fixed1_K8_i2', 2: 'fixed2_K16_i2', 4: 'fixed4'}, 1152: {1: 'fixed1_K16_i2', 2: 'fixed2', 4: 'fixed4_K64_i2'}, 2304: {1: 'fixed1', 2: 'fixed2_K64_i2', 4: 'fixed4_K128_i2'}}
EQ_ROUNDS = {'4 rounds, 64 cand.': 'fixed4_K64_i4', '8 rounds, 32 cand.': 'fixed4_K32_i8'}
KEYS = ('success', 'steps_stop', 'replans_stop', 'units_stop', 'units35', 'replans35', 'best_cov35', 'best_cov_stop', 'eligible_stop', 'nonper_stop')


def load(mode):
    """dict[(arm, damping)] -> dict[key] -> array (seeds, tasks); control arms are averaged over their timing draws."""
    dirs = sorted(RES.glob(f'{mode}_s*'), key=lambda p: int(p.name.split('_s')[1])); seeds = [int(p.name.split('_s')[1]) for p in dirs]; out = {}; metas = []
    for si, d in enumerate(dirs):
        metas.append(json.loads((d / 'meta.json').read_text())); acc = {}
        for r in csv.DictReader(open(d / 'episodes.csv')):
            k = (r['arm'], f"{float(r['damping']):g}"); a = acc.setdefault(k, {}); t = int(r['task'])
            a.setdefault(t, []).append([float(r[x]) for x in KEYS])
        for k, a in acc.items():
            T = max(a) + 1; m = np.array([np.mean(a[t], 0) for t in range(T)])
            o = out.setdefault(k, {x: np.full((len(dirs), T), np.nan) for x in KEYS})
            for j, x in enumerate(KEYS): o[x][si] = m[:, j]
    return out, seeds, metas


def boot_idx(S, T): return RNG.integers(0, S, (B, S)), RNG.integers(0, T, (B, T))


def ci_diff(d, idx, scale):
    si, ti = idx; stat = np.array([d[np.ix_(si[b], ti[b])].mean() for b in range(B)]) * scale
    p = max(1 / B, 2 * min(np.mean(stat <= 0), np.mean(stat >= 0)))
    return dict(diff=float(d.mean() * scale), lo=float(np.quantile(stat, .025)), hi=float(np.quantile(stat, .975)), p=float(min(1., p)))


def rate(D, arm, d='0', si=None, ti=None):
    x = D[(arm, d)]; r, s = x['replans_stop'], x['steps_stop']
    if si is not None: r = r[np.ix_(si, ti)]; s = s[np.ix_(si, ti)]
    return float(r.sum() / s.sum())


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.
    for rank, i in enumerate(order): run = max(run, (len(ps) - rank) * ps[i]); adj[i] = min(1., run)
    return adj.tolist()


def analyse(mode):
    D, seeds, metas = load(mode); S, T = D[('fixed1', '0')]['success'].shape; idx = boot_idx(S, T); out = dict(seeds=seeds, n_seeds=S, tasks=T); N = T
    def arm_stats(arm, d='0'):
        x = D[(arm, d)]; per_seed = x['success'].sum(1)
        return dict(mean=float(per_seed.mean()), min=float(per_seed.min()), max=float(per_seed.max()), sd=float(per_seed.std(ddof=1)) if S > 1 else 0., per_seed=[round(float(v), 2) for v in per_seed],
                    rate=rate(D, arm, d), units_step=rate(D, arm, d) * UNITS * (1 if '_K' not in arm else int(arm.split('_K')[1].split('_')[0]) * int(arm.split('_i')[1]) / 64),
                    units35=float(x['units35'].mean()), units_stop=float(x['units_stop'].mean()), best_cov=float(x['best_cov35'].mean()),
                    s90=float((x['best_cov35'] > .90).sum(1).mean()), s97=float((x['best_cov35'] > .97).sum(1).mean()),
                    trigger_rate=float(x['nonper_stop'].sum() / max(1., x['eligible_stop'].sum())), signal_share=float(x['nonper_stop'].sum() / max(1., x['replans_stop'].sum())), rate35=float(x['replans35'].mean() / 35), best_cov_stop=float(x['best_cov_stop'].mean()))
    out['arms'] = {a: arm_stats(a) for a in FIXED + RULES + ['random~' + r for r in RULES] + [v for g in EQ.values() for v in g.values()] + list(EQ_ROUNDS.values())}
    suc = lambda a, d='0': D[(a, d)]['success']; cov = lambda a, d='0': D[(a, d)]['best_cov35']
    # primary family
    fam = {'F': ci_diff(suc('fixed1') - suc('fixed4'), idx, N)}
    for r in ('error_q75', 'disagreement_q75', 'contact_q75', 'geom'): fam['T-' + r.split('_')[0]] = ci_diff(suc(r) - suc('random~' + r), idx, N)
    adj = holm([fam[k]['p'] for k in fam])
    for k, a in zip(fam, adj): fam[k]['p_holm'] = a
    out['primary'] = fam
    # every rule arm against its matched control and against the fixed-period curve
    si, ti = idx; sweep = {}
    # fixed-period curve, interpolated per seed at an arm's own replans per executed step (analysis_plan.md)
    ar = np.arange(B)[:, None]
    def sums(arm, key): return D[(arm, '0')][key][:, ti].sum(2)                       # (S, B): per seed, per resample of tasks
    def full(arm, key): return D[(arm, '0')][key].sum(1)                              # (S,)
    def knots(f):
        X = np.stack([f(a, 'replans_stop') / f(a, 'steps_stop') for a in FIXED], -1); Y = np.stack([f(a, 'success') / T for a in FIXED], -1); o = np.argsort(X, -1)
        return np.take_along_axis(X, o, -1), np.take_along_axis(Y, o, -1)
    def lerp(q, X, Y):
        i = np.clip((q[..., None] >= X).sum(-1) - 1, 0, X.shape[-1] - 2)[..., None]
        x0 = np.take_along_axis(X, i, -1)[..., 0]; x1 = np.take_along_axis(X, i + 1, -1)[..., 0]; y0 = np.take_along_axis(Y, i, -1)[..., 0]; y1 = np.take_along_axis(Y, i + 1, -1)[..., 0]
        return y0 + np.clip((q - x0) / (x1 - x0), 0, 1) * (y1 - y0)
    KB = knots(sums); KF = knots(full)
    def curve_diff(arm):
        d = sums(arm, 'success') / T - lerp(sums(arm, 'replans_stop') / sums(arm, 'steps_stop'), *KB)        # (S, B)
        stat = d[si, ar].mean(1) * N; pt = float((full(arm, 'success') / T - lerp(full(arm, 'replans_stop') / full(arm, 'steps_stop'), *KF)).mean() * N)
        return dict(diff=pt, lo=float(np.quantile(stat, .025)), hi=float(np.quantile(stat, .975)), p=float(min(1., max(1 / B, 2 * min(np.mean(stat <= 0), np.mean(stat >= 0))))))
    for r in RULES:
        c = 'random~' + r; e = dict(vs_control=ci_diff(suc(r) - suc(c), idx, N), vs_control_cov=ci_diff(cov(r) - cov(c), idx, 1.), rule_rate=rate(D, r), control_rate=rate(D, c),
                                   matched=[bool(m['matching'][r]['matched']) for m in metas], p_control=[round(m['matching'][r]['p'], 3) for m in metas])
        e['vs_curve'] = curve_diff(r); e['control_vs_curve'] = curve_diff(c)
        sweep[r] = e
    for r, a in zip(RULES, holm([sweep[r]['vs_control']['p'] for r in RULES])): sweep[r]['vs_control']['p_holm13'] = a      # exploratory correction across all 13 rule settings
    out['sweep'] = sweep
    out['thresholds'] = {s: {q: [m['thresholds'][s][str(q)] for m in metas] for q in QS} for s in SCORES}
    # additional contrasts specified in the analysis plan
    pairs = {'fixed1.fixed2': ('fixed1', 'fixed2'), 'fixed2.fixed3': ('fixed2', 'fixed3'), 'fixed3.fixed4': ('fixed3', 'fixed4'), 'fixed4.fixed6': ('fixed4', 'fixed6'),
             'period.576': (EQ[576][1], EQ[576][4]), 'period.1152': (EQ[1152][1], EQ[1152][4]), 'period.2304': (EQ[2304][1], EQ[2304][4]),
             'budget.p1': (EQ[2304][1], EQ[576][1]), 'budget.p2': (EQ[2304][2], EQ[576][2]), 'budget.p4': (EQ[2304][4], EQ[576][4]),
             # exploratory contrasts among specified arms
             'rounds.r8.fixed1': ('fixed4_K32_i8', 'fixed1'), 'rounds.r4.fixed1': ('fixed4_K64_i4', 'fixed1'), 'rounds.r8.k128': ('fixed4_K32_i8', 'fixed4_K128_i2'), 'rounds.r4.k128': ('fixed4_K64_i4', 'fixed4_K128_i2'),
             'rounds.r8.fixed4': ('fixed4_K32_i8', 'fixed4'), 'cand.k64.fixed4': ('fixed4_K64_i2', 'fixed4'), 'cand.k128.k64': ('fixed4_K128_i2', 'fixed4_K64_i2')}
    out['pairs'] = {k: dict(success=ci_diff(suc(a) - suc(b), idx, N), cov=ci_diff(cov(a) - cov(b), idx, 1.)) for k, (a, b) in pairs.items()}
    pc = {'F': ci_diff(cov('fixed1') - cov('fixed4'), idx, 1.)}          # the same five comparisons on the continuous outcome
    for r in ('error_q75', 'disagreement_q75', 'contact_q75', 'geom'): pc['T-' + r.split('_')[0]] = ci_diff(cov(r) - cov('random~' + r), idx, 1.)
    for k, a in zip(pc, holm([pc[k]['p'] for k in pc])): pc[k]['p_holm'] = a
    out['primary_cov'] = pc
    both = [(fam, k) for k in fam] + [(pc, k) for k in pc]          # exploratory Holm correction across ten comparisons on both outcomes
    for (d_, k), a in zip(both, holm([d_[k]['p'] for d_, k in both])): d_[k]['p_holm10'] = a
    for r, a in zip(RULES, holm([sweep[r]['vs_control_cov']['p'] for r in RULES])): sweep[r]['vs_control_cov']['p_holm13'] = a
    out['primary_sens'] = {str(th): ci_diff((cov('fixed1') > th).astype(float) - (cov('fixed4') > th).astype(float), idx, N) for th in (.90, .97)}
    out['equal'] = {str(u): {str(p): out['arms'][a]['mean'] for p, a in g.items()} for u, g in EQ.items()}; out['equal_rounds'] = {k: out['arms'][a]['mean'] for k, a in EQ_ROUNDS.items()}
    # unseen damping
    sh = {}
    for d in SHIFT:
        sh[d] = {a: arm_stats(a, d) for a in ('fixed1', 'fixed4', 'error_q75')}
        x = D[('fixed1', d)]; sh[d]['fixed1']['mean_best_cov'] = float(x['best_cov35'].mean())
    out['shift'] = sh
    out['timing'] = [m.get('timing') for m in metas]; out['matching_unmatched'] = {r: int(sum(not m['matching'][r]['matched'] for m in metas)) for r in RULES}
    return out


def oracle():
    d = RES / 'oracle_s0'
    if not (d / 'meta.json').exists(): return None
    rows = list(csv.DictReader(open(d / 'episodes.csv'))); meta = json.loads((d / 'meta.json').read_text()); out = dict(coast=meta['coast_px'], arms={})
    for r in rows:
        k = f"{r['arm']}@{float(r['damping']):g}"; o = out['arms'].setdefault(k, dict(success=0, n=0, cov=[])); o['success'] += int(r['success']); o['n'] += 1; o['cov'].append(float(r['best_cov_stop']))
    for o in out['arms'].values(): o['cov'] = float(np.mean(o['cov']))
    per = {}
    for r in rows:
        if float(r['damping']) == 0: per.setdefault(r['arm'], {})[int(r['task'])] = int(r['success'])
    T = len(per['fixed1']); ti = RNG.integers(0, T, (B, T)); out['pairs'] = {}
    for k, (a, b) in {'fixed1.fixed4': ('fixed1', 'fixed4'), 'fixed1.fixed6': ('fixed1', 'fixed6'), 'fixed1.fixed2': ('fixed1', 'fixed2')}.items():
        d = np.array([per[a][t] - per[b][t] for t in range(T)], float); stat = d[ti].sum(1)
        out['pairs'][k] = dict(diff=float(d.sum()), lo=float(np.quantile(stat, .025)), hi=float(np.quantile(stat, .975)), p=float(min(1., max(1 / B, 2 * min(np.mean(stat <= 0), np.mean(stat >= 0))))))
    return out


if __name__ == '__main__':
    res = {}
    for mode in ('state', 'pixels'):
        if list(RES.glob(f'{mode}_s*/meta.json')): res[mode] = analyse(mode)
    res['oracle'] = oracle()
    (HERE / f'{PREFIX}stage2.json').write_text(json.dumps(res, indent=1))
    for mode in ('state', 'pixels'):
        if mode not in res: continue
        r = res[mode]; print(f"== {mode}: seeds {r['seeds']}")
        for a in FIXED: print(f"  {a:16s} {r['arms'][a]['mean']:5.1f} [{r['arms'][a]['min']:.0f}-{r['arms'][a]['max']:.0f}] rate {r['arms'][a]['rate']:.3f}")
        for k, v in r['primary'].items(): print(f"  {k:16s} diff {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] p {v['p']:.4f} holm {v['p_holm']:.4f}")
        for a in RULES:
            s = r['sweep'][a]; print(f"  {a:18s} {r['arms'][a]['mean']:5.1f} rate {s['rule_rate']:.3f} | ctrl {r['arms']['random~' + a]['mean']:5.1f} rate {s['control_rate']:.3f} | d {s['vs_control']['diff']:+.2f} [{s['vs_control']['lo']:+.2f},{s['vs_control']['hi']:+.2f}] | curve {s['vs_curve']['diff']:+.2f} [{s['vs_curve']['lo']:+.2f},{s['vs_curve']['hi']:+.2f}] | ctrl-curve {s['control_vs_curve']['diff']:+.2f} [{s['control_vs_curve']['lo']:+.2f},{s['control_vs_curve']['hi']:+.2f}] holm13 {s['vs_control']['p_holm13']:.3f} unmatched {r['matching_unmatched'][a]}")
        print('  equal', r['equal'], r['equal_rounds'])
        for k, v in r['pairs'].items(): print(f"  {k:14s} {v['success']['diff']:+.2f} [{v['success']['lo']:+.2f},{v['success']['hi']:+.2f}] p {v['success']['p']:.4f}")
        print('  sens', r['primary_sens'])
        for k, v in r['primary_cov'].items(): print(f"  cov {k:14s} {v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] p {v['p']:.4f} holm {v['p_holm']:.4f}")
        for a in RULES: v = r['sweep'][a]['vs_control_cov']; print(f"  covsweep {a:18s} {v['diff']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] p {v['p']:.4f} holm13 {v['p_holm13']:.4f}")
        for d in SHIFT: print(f"  d={d}: fixed1 {r['shift'][d]['fixed1']['mean']:.1f} fixed4 {r['shift'][d]['fixed4']['mean']:.1f} error trig {r['shift'][d]['error_q75']['trigger_rate']:.2f}")
    print(res['oracle'])
