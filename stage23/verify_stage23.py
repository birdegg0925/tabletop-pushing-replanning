#!/usr/bin/env python3
"""Checks on the Stage 2 and Stage 3 package. Run from this folder: python verify_stage23.py [--quick] [--prefix local_]

1. files: every model and result file is present and has the expected number of episodes;
2. construction: every schedule-shuffled control episode replays the schedule of another task of the same rule, each used once per draw;
3. analysis: rev_analysis.py and rev_analysis3.py, run on the result files, reproduce stage2.json and stage3.json;
4. replay: a few rule and control episodes are run again with the locally generated models and compared step by step with the stored rows.

The file, schedule-construction and analysis checks determine all_passed.
Selected replay comparisons are reported separately because floating-point results can vary by environment.
The quick check uses the pilot model and skips reanalysis."""
import csv, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; QUICK = '--quick' in sys.argv
import rev_eval as E

PREFIX = E.PREFIX; RES = HERE / f'{PREFIX}results'; R3 = HERE / f'{PREFIX}results3'
STATE = [23, 29, 37, 43, 53, 59, 61, 67, 73, 79]; PIXELS = [23, 29, 37, 43, 53]
RULES = [f'{s}_q{q}' for s in E.SCORES for q in E.QS] + ['geom']


def rows_of(path): return list(csv.DictReader(open(path)))


def files():
    out = dict(missing=[], counts={})
    for mode, seeds in (('state', STATE), ('pixels', PIXELS)):
        for s in seeds:
            d = f'{mode}_s{s}'; need = [RES / d / 'episodes.csv', RES / d / 'meta.json', R3 / d / 'shuffle.csv', R3 / d / 'floor.json', E.model_path(mode, s)] + ([R3 / d / 'mild.csv'] if mode == 'state' else [])
            out['missing'] += [str(p.relative_to(HERE.parent)) for p in need if not p.exists()]
    for p in (RES / 'oracle_s0' / 'episodes.csv', R3 / 'oracle_s0' / 'mild.csv'):
        if not p.exists(): out['missing'].append(str(p.relative_to(HERE.parent)))
    if not out['missing']:
        n2 = len(rows_of(RES / 'state_s23' / 'episodes.csv')); n3 = len(rows_of(R3 / 'state_s23' / 'shuffle.csv')); out['counts'] = dict(stage2_rows_pilot=n2, shuffle_rows_pilot=n3)
        out['counts_ok'] = n3 == len(RULES) * 20 * E.TASKS
    out['ok'] = not out['missing'] and out.get('counts_ok', False); return out


def construction(mode, seed):
    rule = {}
    for r in rows_of(RES / f'{mode}_s{seed}' / 'episodes.csv'):
        if r['arm'] in RULES and float(r['damping']) == 0 and r['draw'] == '0': rule[(r['arm'], int(r['task']))] = r['replan_bits']
    bad = 0; donors = {}
    for r in rows_of(R3 / f'{mode}_s{seed}' / 'shuffle.csv'):
        a = r['arm'][8:]; t = int(r['task']); d = int(r['donor']); donors.setdefault((a, r['draw']), []).append(d)
        bad += int(d == t or r['replan_bits'] != rule[(a, d)])
    bad += sum(sorted(v) != list(range(E.TASKS)) for v in donors.values())
    return dict(model=f'{mode}_s{seed}', control_episodes=sum(len(v) for v in donors.values()), violations=bad)


def close(a, b, tol=1e-9):
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list): return isinstance(b, list) and len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float): return a is not None and b is not None and abs(a - b) <= tol * max(1., abs(a), abs(b))
    return a == b


def analysis():
    import rev_analysis as A2, rev_analysis3 as A3
    res2 = {m: A2.analyse(m) for m in ('state', 'pixels')}; res2['oracle'] = A2.oracle()
    res3 = {m: A3.analyse(m) for m in ('state', 'pixels')}; res3['mild'] = A3.mild(); res3['floor'] = A3.floors()
    ref2 = json.loads((HERE / f'{PREFIX}stage2.json').read_text()); ref3 = json.loads((HERE / f'{PREFIX}stage3.json').read_text())
    j = lambda x: json.loads(json.dumps(x))
    return dict(stage2_reproduced=close(j(res2), ref2), stage3_reproduced=close(j(res3), ref3),
                headline=dict(every_step=res2['state']['arms']['fixed1']['mean'], every_4=res2['state']['arms']['fixed4']['mean'],
                              rule_minus_shuffled_error_q50=res3['state']['sweep']['error_q50']['rule_shuffle']['diff'], shuffled_minus_rate_matched_error_q50=res3['state']['sweep']['error_q50']['shuffle_random']['diff']))


def replay(mode, seed, tasks=(0, 1), arms=('error_q50', 'contact_q75', 'geom')):
    meta = json.loads((RES / f'{mode}_s{seed}' / 'meta.json').read_text()); model = str(E.model_path(mode, seed)); stored = {}; shuf = {}
    for r in rows_of(RES / f'{mode}_s{seed}' / 'episodes.csv'):
        if r['arm'] in RULES and float(r['damping']) == 0 and r['draw'] == '0': stored[(r['arm'], int(r['task']))] = r
    for r in rows_of(R3 / f'{mode}_s{seed}' / 'shuffle.csv'):
        if r['draw'] == '1' and int(r['task']) in tasks: shuf[(r['arm'][8:], int(r['task']))] = r
    jobs = []
    for t in tasks:
        items = []
        for a in arms:
            s = a.split('_')[0]; cal = {} if a == 'geom' else {s + '_threshold': meta['thresholds'][s][a.split('_q')[1]]}
            items.append((0., dict(name=a, method=s, cal=cal), [0], False))
            items.append((0., dict(name='shuffle~' + a, method='schedule', bits=stored[(a, int(shuf[(a, t)]['donor']))]['replan_bits']), [1], False))
        jobs.append(('ep', t, E.TEST_SEED, items))
    same = total = 0
    for r in E.pool_run(model, jobs, 2):
        ref = shuf[(r['arm'][8:], r['task'])] if r['arm'].startswith('shuffle~') else stored[(r['arm'], r['task'])]
        total += 1; same += int(r['cov_series'] == ref['cov_series'] and r['replan_bits'] == ref['replan_bits'] and int(r['success']) == int(ref['success']))
    return dict(model=f'{mode}_s{seed}', episodes=total, identical=same)


if __name__ == '__main__':
    t0 = time.time(); out = dict(prefix=PREFIX, quick=QUICK, files=files())
    if out['files']['ok']:
        models = [('state', 23)] if QUICK else [('state', s) for s in STATE] + [('pixels', s) for s in PIXELS]
        out['construction'] = [construction(m, s) for m, s in models]
        if not QUICK: out['analysis'] = analysis()
        out['replay'] = [replay(m, s) for m, s in ([('state', 23)] if QUICK else [('state', 23), ('state', 29), ('pixels', 23)])]
    out['construction_ok'] = all(c['violations'] == 0 for c in out.get('construction', [])) and bool(out.get('construction'))
    out['analysis_ok'] = QUICK or (out.get('analysis', {}).get('stage2_reproduced', False) and out.get('analysis', {}).get('stage3_reproduced', False))
    out['replay_exact'] = all(r['identical'] == r['episodes'] for r in out.get('replay', [])) and bool(out.get('replay'))
    out['all_passed'] = bool(out['files']['ok'] and out['construction_ok'] and out['analysis_ok']); out['seconds'] = round(time.time() - t0)
    (HERE / f"{PREFIX}verification_stage23{'_quick' if QUICK else ''}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))      # the quick check never replaces the full record
    print(json.dumps({k: out[k] for k in ('all_passed', 'construction_ok', 'analysis_ok', 'replay_exact', 'seconds')})); print(json.dumps(out.get('replay')), json.dumps(out.get('analysis', {}).get('headline')))
    sys.exit(0 if out['all_passed'] else 1)
