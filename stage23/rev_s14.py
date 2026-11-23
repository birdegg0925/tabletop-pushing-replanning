#!/usr/bin/env python3
"""Compare near-contact and execution-error trigger rates across calibration and test tasks.
The exploratory diagnostic uses the 75th-percentile threshold and 48 calibration tasks per state-input seed.
Test rates come from Stage 2 episode files.
The filenames rev_s14.py and s14.json identify this calibration-to-test diagnostic."""
import csv, json, sys
from pathlib import Path
import numpy as np
import rev_eval as E

HERE = Path(__file__).resolve().parent; RES = HERE / f'{E.PREFIX}results'

if __name__ == '__main__':
    out = {}
    for d in sorted(RES.glob('state_s*'), key=lambda p: int(p.name.split('_s')[1])):
        meta = json.loads((d / 'meta.json').read_text()); seed = int(d.name.split('_s')[1]); res = {}
        for s in ('contact', 'error'):
            arm = dict(name=f'{s}_q75', method=s, cal={s + '_threshold': meta['thresholds'][s]['75']})
            rows = E.pool_run(str(E.model_path('state', seed)), [('ep', t, E.CAL_SEED, [(0., arm, [0], False)]) for t in range(E.CAL_TASKS)], 12)
            cal = sum(r['nonper_stop'] for r in rows) / max(1, sum(r['eligible_stop'] for r in rows)); el = fi = 0
            for r in csv.DictReader(open(d / 'episodes.csv')):
                if r['arm'] == arm['name'] and float(r['damping']) == 0: el += int(r['eligible_stop']); fi += int(r['nonper_stop'])
            res[s] = dict(cal=cal, test=fi / el, cal_success=sum(r['success'] for r in rows) / len(rows))
        out[seed] = res; print(seed, json.dumps(res), flush=True)
    agg = {s: dict(cal=float(np.mean([out[k][s]['cal'] for k in out])), test=float(np.mean([out[k][s]['test'] for k in out])),
                   cal_higher=int(sum(out[k][s]['cal'] > out[k][s]['test'] for k in out)), n=len(out)) for s in ('contact', 'error')}
    (HERE / f'{E.PREFIX}s14.json').write_text(json.dumps(dict(per_seed=out, mean=agg), indent=1)); print(json.dumps(agg))
