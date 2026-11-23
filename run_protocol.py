#!/usr/bin/env python3
"""Generate or run the pilot and full PushT experiment configurations."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN_UNITS = 3 * 32 * 12 * 2        # members x candidates x (6 free + 6 tail steps) x 2 search rounds = 2304


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', choices=['pilot', 'full'], default='pilot')
    p.add_argument('--root', default='.', help='directory that receives data/, models/, results/, audits/, figures/ and logs/')
    p.add_argument('--prefix', default='', help='name prefix for every output, for example local_')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='device for the image front-end')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dampings', type=float, nargs='+', help='evaluation dampings; full execution is limited to conditions the gate report lists as eligible')
    p.add_argument('--gate-report', help='gate_report.json produced by run.py gate; required for full execution')
    p.add_argument('--override-gate', action='store_true', help='run full despite an unmet gate; the protocol file records this override')
    p.add_argument('--execute', action='store_true', help='run the commands; omitted means write and print the reviewable plan only')
    a = p.parse_args(); root = Path(a.root); P = a.prefix; full = a.profile == 'full'
    dampings = a.dampings or [0.0, 0.25, 0.85]
    protocol = dict(profile=a.profile, collection_seed=117 if full else 19, exploration_episodes=2400, episode_steps=32, policy_rounds=3, policy_episodes_per_round=1200,
                    training_dampings=[0.0], training_seeds=[23, 29, 37, 43, 53] if full else [23], modes=['state', 'pixels'], dynamics_epochs=60, vision_epochs=30,
                    calibration_task_seed=152 if full else 52, calibration_tasks=48, test_task_seed=141 if full else 71, tasks_per_condition=64, dampings=dampings,
                    budgets=[0, 12 * PLAN_UNITS, 18 * PLAN_UNITS] if full else [0, 12 * PLAN_UNITS], plan_units=PLAN_UNITS,
                    methods=['fixed1', 'fixed2', 'fixed3', 'fixed4', 'fixed6', 'error', 'disagreement', 'contact', 'geom', 'random'], horizon=6, tail=6, candidates=32, search_rounds=2, max_commit=6,
                    device=a.device, workers=a.workers, prefix=P, gate_report=a.gate_report, gate_override=a.override_gate,
                    reference_code_sha256=hashlib.sha256((HERE / 'run.py').read_bytes()).hexdigest(),
                    caveats=['This configuration records commands. Execution is recorded in the run logs.',
                             'Training uses damping 0. Other damping values test dynamics shift.',
                             'Each training seed identifies a separate model fit.',
                             'Budget 0 is uncapped. Capped runs use separate output directories.'])
    plan = []
    def run(*cmd): plan.append([sys.executable, str(HERE / 'run.py'), *map(str, cmd)])
    D = lambda n: root / 'data' / f'{P}{n}' / 'dataset.npz'; M = lambda n: root / 'models' / f'{P}{n}' / 'model.pt'
    R = lambda n: root / 'results' / f'{P}{n}'; A = lambda n: root / 'audits' / f'{P}{n}'; W = ['--workers', a.workers]
    tag = lambda b: 'unlimited' if b == 0 else f'b{b // PLAN_UNITS}'
    run('collect', '--out', D('explore').parent, '--episodes', protocol['exploration_episodes'], '--seed', protocol['collection_seed'], '--images')
    run('audit', '--data', D('explore'), '--out', A('initial'))
    for seed in protocol['training_seeds']:
        s = '' if not full else f'_seed{seed}'; data = [D('explore')]
        run('train', '--data', *data, '--out', M(f'state_r0{s}').parent, '--mode', 'state', '--seed', seed)
        for r in range(1, protocol['policy_rounds'] + 1):       # on-policy aggregation: the newest controller collects, the next model trains on everything
            run('collect-policy', '--model', M(f'state_r{r - 1}{s}'), '--out', D(f'policy_r{r}{s}').parent, '--episodes', protocol['policy_episodes_per_round'],
                '--seed', r if not full else seed * 10 + r, '--images', *W)
            data.append(D(f'policy_r{r}{s}')); name = f'state_pilot{s}' if r == protocol['policy_rounds'] else f'state_r{r}{s}'
            run('train', '--data', *data, '--out', M(name).parent, '--mode', 'state', '--seed', seed)
        run('train', '--data', *data, '--out', M(f'pixels_pilot{s}').parent, '--mode', 'pixels', '--seed', seed, '--vision-epochs', protocol['vision_epochs'], '--device', a.device)
        for mode in protocol['modes']:
            model = M(f'{mode}_pilot{s}')
            run('calibrate', '--model', model, '--tasks', protocol['calibration_tasks'], '--seed', protocol['calibration_task_seed'])
            run('audit', '--data', *data, '--model', model, '--out', A(f'{mode}_pilot{s}'))
            for d in dampings:
                for b in (protocol['budgets'] if d == 0 else [0]):
                    out = R(f'{mode}_d{d}_{tag(b)}{s}')
                    run('evaluate', '--model', model, '--out', out, '--tasks', protocol['tasks_per_condition'], '--seed', protocol['test_task_seed'], '--damping', d, '--budget', b,
                        '--gif', 1 if (d == 0 and b == 0) else 0, *W)
                    run('report', '--input', out, '--out', root / 'figures' / out.name)
    if not full:
        for mode in protocol['modes']:      # the eight tasks of the first-version pilot, kept for a before/after reading only
            for d in dampings: run('evaluate', '--model', M(f'{mode}_pilot'), '--out', R(f'legacy41_{mode}_d{d}'), '--tasks', 8, '--seed', 41, '--damping', d, *W)
        for d in [0.0, 0.25]:               # Score the same search with simulator rollouts.
            run('oracle', '--out', R(f'oracle_d{d}'), '--tasks', 8, '--seed', 51, '--damping', d, *W)
        for r in range(4):                  # development tasks: what each aggregation round changes; the last round runs the whole fixed-frequency grid
            run('evaluate', '--model', M('state_pilot' if r == 3 else f'state_r{r}'), '--out', R(f'dev51_round{r}'), '--tasks', 16, '--seed', 51, '--damping', 0,
                '--methods', 'fixed1,fixed2,fixed3,fixed4,fixed6' if r == 3 else 'fixed1,fixed4', *W)
        # ablation with equal data size: exploration data mixing three dampings, compared with dev51_round0 (damping 0 only)
        run('collect', '--out', D('explore_mixed').parent, '--episodes', protocol['exploration_episodes'], '--seed', 21, '--dampings', 0, 0.25, 0.6)
        run('train', '--data', D('explore_mixed'), '--out', M('state_mixed').parent, '--mode', 'state', '--seed', 23)
        run('evaluate', '--model', M('state_mixed'), '--out', R('dev51_mixed_damping'), '--tasks', 16, '--seed', 51, '--damping', 0, '--methods', 'fixed1,fixed4', *W)
        plan.append([sys.executable, str(HERE / 'verify.py'), '--root', str(root), '--prefix', P, '--out', str(root / 'audits' / f'{P}final_verification.json')])
        run('gate', '--results', root / 'results', '--prefix', P, '--audit', A('state_pilot') / 'audit.json', '--verification', root / 'audits' / f'{P}final_verification.json',
            '--out', root / f'{P}gate_report.json')
    root.mkdir(parents=True, exist_ok=True); base = root if full else root / 'logs'; base.mkdir(parents=True, exist_ok=True)
    (base / f'{P}protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    (base / f'{P}commands.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f'profile={a.profile}; commands={len(plan)}; execute={a.execute}', flush=True)
    if not a.execute:
        print(f'Plan only: {base / f"{P}commands.json"}', flush=True); return
    if full and not a.override_gate:
        gate = {} if not a.gate_report else json.loads(Path(a.gate_report).read_text())
        blocked = [d for d in dampings if d not in gate.get('eligible_conditions', [])]
        if blocked:
            raise SystemExit(f'Full execution blocked: the baseline gate has not passed for dampings {blocked}. Restrict --dampings to eligible conditions listed in gate_report.json, '
                             'or improve the pilot first. An explicit --override-gate records deliberate exploratory execution.')
    logdir = root / 'logs'; logdir.mkdir(exist_ok=True)
    for i, cmd in enumerate(plan):
        name = Path(cmd[1]).stem if Path(cmd[1]).name != 'run.py' else cmd[2]
        print(f'command {i + 1}/{len(plan)}: {name}', flush=True)
        with (logdir / f'{P}{i + 1:03d}_{name}.log').open('w') as log:
            done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if done.returncode: raise SystemExit(f'Command failed; inspect {logdir / f"{P}{i + 1:03d}_{name}.log"}')


if __name__ == '__main__': main()
