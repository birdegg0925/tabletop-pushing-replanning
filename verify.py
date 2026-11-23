#!/usr/bin/env python3
"""Behavioural checks: privileged-state isolation, budget accounting, trigger logs, simulator semantics, checkpoint-format rejection."""
import argparse, csv, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
from run import Agent, Budget, Scene, explore_action, goal_task, save_json, t_sdf, trigger_diagnostics

HERE = Path(__file__).resolve().parent


def checkpoint_format_checks():
    """Reject synthetic incompatible files without requiring archived model weights."""
    checks = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, payload in [('unversioned', {}), ('v0', {'format': 'v0'}), ('v1', {'format': 'v1'})]:
            path = Path(tmp) / f'{label}.pt'
            torch.save(payload, path)
            try:
                Agent(path)
                rejected = False
            except RuntimeError as exc:
                rejected = 'Checkpoint format mismatch' in str(exc)
            checks[f'synthetic_{label}_checkpoint_rejected'] = rejected
    return checks


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', default='.'); p.add_argument('--prefix', default=''); p.add_argument('--out', default='audits/final_verification.json')
    a = p.parse_args(); root = Path(a.root); P = a.prefix; checks = {}
    agent = Agent(root / 'models' / f'{P}pixels_pilot' / 'model.pt'); scene, goal = goal_task(942001, 0., agent.image_size)
    def forbidden(): raise RuntimeError('privileged scene.obs disabled')
    scene.obs = forbidden
    # Only the image observation and the specified goal reach the deployed control code.
    previous = agent.observe(scene); current = previous.copy(); units = agent.E * 8 * (4 + 4) * 2; budget = Budget(units)
    plan = agent.plan(previous, current, goal, budget, np.random.default_rng(711), current[:2], horizon=4, candidates=8, iters=2, warm=None, tail=4)
    scene.env.step(plan['actions'][0]); nxt = agent.observe(scene)
    diagnostic = trigger_diagnostics('contact', 1, plan, nxt, agent.cal, 6, np.random.default_rng(1))
    checks['visual_observe_plan_trigger_without_privileged_scene_obs'] = bool(np.isfinite(nxt).all() and isinstance(diagnostic['trigger'], bool))
    checks['plan_charges_members_x_candidates_x_steps_x_rounds'] = budget.used == units
    checks['budget_exhaustion_does_not_call_predictor'] = agent.plan(current, nxt, goal, budget, np.random.default_rng(711), nxt[:2], 4, 8, 2, None, 4) is None and budget.used == budget.limit
    scene.close()
    errors = []
    for seed in range(12):        # the analytic T-outline distance used by the model and the geometry trigger matches the simulator
        s = Scene(1000 + seed, 0.); rng = np.random.default_rng(seed)
        for t in range(10):
            s.step(explore_action(s, rng, t, seed % 4)); errors.append(abs(t_sdf(s.obs()) - min(sh.point_query(s.env.agent.position).distance for sh in s.env.block.shapes)))
        s.close()
    checks['outline_distance_matches_simulator'] = bool(max(errors) < 1e-3)
    coast = {}
    for d in [0., .25]:           # damping 0 stops the block at release; larger values let it keep sliding
        s = Scene(5, d)
        for t in range(4): pp = s.obs()[:2] * 512; b = s.obs()[2:4] * 512; s.step(pp + (b - pp) / np.linalg.norm(b - pp) * 40)
        hold = s.obs()[:2] * 512; b0 = s.obs()[2:4] * 512
        for t in range(10): s.step(hold)
        coast[d] = float(np.linalg.norm(s.obs()[2:4] * 512 - b0)); s.close()
    checks['damping_zero_stops_block_and_larger_values_slide'] = coast[0.] < 10. and coast[.25] > 40.
    checks.update(checkpoint_format_checks())
    rows_checked = 0; runs = sorted(q.parent for q in (root / 'results').glob(f'{P}*/episodes.csv') if 'oracle' not in q.parent.name)
    for run in runs:
        rows = list(csv.DictReader(open(run / 'episodes.csv'))); traces = list(csv.DictReader(open(run / 'traces.csv'))); name = run.name
        cap = int(rows[0]['budget'])
        checks[f'{name}_budget_all_rows'] = all(cap == 0 or int(r['model_calls']) <= cap for r in rows)
        checks[f'{name}_trigger_logging'] = all(k in traces[0] for k in ['age_before', 'budget_before', 'budget_after', 'trigger_reason', 'score_error', 'score_disagreement', 'score_contact', 'score_geom'])
        checks[f'{name}_budget_monotonic'] = all(int(r['budget_before']) <= int(r['budget_after']) and (cap == 0 or int(r['budget_after']) <= cap) for r in traces)
        rows_checked += len(rows)
    nominal = root / 'results' / f'{P}state_d0.0_unlimited' / 'summary.json'
    if nominal.exists(): checks['all_schedules_behave_differently_at_nominal'] = all(len(g) == 1 for g in json.loads(nominal.read_text())['identical_behaviour_groups'])
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, str(HERE / 'run_protocol.py'), '--profile', 'full', '--root', tmp, '--execute'], capture_output=True, text=True)
        checks['full_protocol_blocked_without_passed_gate'] = r.returncode != 0 and 'Full execution blocked' in r.stderr
    payload = {'checks': checks, 'all_passed': all(checks.values()), 'checks_total': len(checks), 'result_directories_checked': len(runs), 'episode_rows_checked': rows_checked,
               'coast_px_10_steps_after_release': {str(k): v for k, v in coast.items()}, 'scope': 'Simulator, model and episode consistency checks.'}
    save_json(a.out, payload); print(json.dumps(payload, ensure_ascii=False))
    if not payload['all_passed']: raise SystemExit(1)


if __name__ == '__main__': main()
