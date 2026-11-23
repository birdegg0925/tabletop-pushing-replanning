# Stage 2 analysis plan

Recorded on 18 November 2026, before the Stage 2 reference simulations.

Stage 1 is the existing pilot: one training seed (23), 64 test tasks (task seed 71). Its numbers stay as reported and are treated as exploratory. Everything below was fixed before looking at any Stage 2 outcome.

## Code

The reference implementation ([run.py](../run.py)) is imported unchanged. [rev_eval.py](rev_eval.py) adds its own episode loop so that an episode can run for the full 35 steps, a timing rule can be given any threshold, the random control can be given any probability and timing draw, and the candidate count and number of search rounds can be set per arm. A self-check requires the new loop to reproduce `run.py`'s `run_episode` exactly (success, steps, units, replans, best coverage) for all ten pilot schedules, and the 75th-percentile thresholds to equal the pilot's `calibration.json`.

## Models

- State input: ten training seeds, 23 (the pilot model) and 29, 37, 43, 53, 59, 61, 67, 73, 79.
- Image input: five training seeds, 23 (the pilot model) and 29, 37, 43, 53; image front-end trained for 30 epochs, as in the pilot's final configuration.
- All seeds share the pilot's exploration set (collection seed 19). Each seed has its own three on-policy rounds (collection seed 10 x seed + round) and its own ensemble initialisation and bootstrap. Seeds therefore vary initialisation, bootstrap resampling and on-policy data; they do not vary the exploration set.
- Training condition: damping 0 only.

## Tasks

- Calibration: 48 tasks, task seed 52, per model (as in the pilot).
- Test: 64 tasks, task seed 83. The same 64 held-out tasks are used for every model, so seeds and tasks are crossed.
- Episodes run for 35 steps. Success is coverage above 0.95 at any step, as in the pilot; the trajectory up to first success is identical to an episode that stops there.

## Arms (damping 0, no cap), per model

1. Fixed periods 1, 2, 3, 4, 6 with 32 candidates and 2 search rounds.
2. Timing rules. Execution error, ensemble disagreement and predicted near-contact change, each at four thresholds: the 25th, 50th, 75th and 90th percentile of that score on the model's own calibration episodes (every-six-steps schedule). The 75th percentile is the pilot's operating point. The current-geometry rule has no threshold and is one arm. Thirteen rule arms.
3. Matched random-timing control, one per rule arm. At each eligible step the control replans with probability p. For a constant p the long-run replans per step is p / (1 - (1 - p)^6); p is first set so that this equals the rule arm's realised replans per executed step on the test tasks (total replans divided by total steps up to first success or step 35). Four tuning draws per task are then run, only their replans per executed step is read, and p is adjusted once by the difference. The reported control is 20 further independent timing draws per task at the adjusted p. Outcomes of the tuning draws are discarded. If the realised rates of rule and control differ by more than 0.015 replans per step the arm is flagged and reported as not matched; p is not tuned again.
4. Equal-budget arms. Period by per-step budget grid, with units per plan = 3 x candidates x 12 x rounds: 576 units per step (period 1 with 8 candidates; period 2 with 16), 1152 (period 1 with 16; period 4 with 64), 2304 (period 2 with 64; period 4 with 128; period 4 with 64 candidates and 4 rounds; period 4 with 32 candidates and 8 rounds). Together with the standard fixed-period arms this gives periods 1, 2, 4 at each of three per-step budgets.

## Other conditions

- Unseen damping: 0.02, 0.05, 0.10, 0.25, 0.85. Arms: every step, every four steps, execution-error rule at the 75th percentile. Goals are generated under each damping, as in the pilot. Coasting distance after release is measured for each value with the audit procedure.
- Simulator-scored planner (exact rollouts, same proposals and cost): fixed periods 1, 2, 3, 4, 6 at damping 0, and every step at each unseen damping; 64 test tasks; episodes stop at first success.
- Timing: milliseconds per full plan (32 candidates, 2 rounds) and per image front-end call, single thread, on the machine used.

## Cost axis

Primary: replans per executed step (and the same multiplied by units per plan, prediction units per executed step), computed over steps up to first success or step 35. For fixed periods and for the random control this is set by the schedule. Secondary: prediction units over the full 35-step episode. The pilot's axis, mean units per task up to first success, is reported as a third column only.

## Outcomes

- Primary: success (coverage above 0.95 at any step within 35).
- Co-primary, continuous: best coverage over the 35 steps.
- Sensitivity: success at thresholds 0.90 and 0.97.

## Planned comparisons

Primary family (state input, five comparisons, Holm correction):

- F: every step against every four steps (success difference).
- T-error, T-disagreement, T-contact: each rule at its 75th-percentile threshold against its matched random control (rule success minus the mean over the control's 20 draws).
- T-geometry: the geometry rule against its matched random control.

Secondary (reported with intervals, uncorrected, labelled as such):

- Each rule arm at the other thresholds against its matched control.
- Each rule arm against the fixed-period curve at the same replans per executed step (linear interpolation between neighbouring fixed periods, per seed).
- Period effect at equal per-step budget: period 1 against period 4 at 576 and at 2304 units per step; budget effect at fixed period: 576 against 2304 units per step at period 1 and at period 4.
- The same comparisons for image input.
- Unseen damping: success of every-step replanning and trigger rate of the execution-error rule at each damping; simulator-scored planner at each damping.

## Inference

- Unit of analysis: (seed, task). Point estimates are means over seeds and tasks; per-seed counts are reported for every arm.
- Intervals: crossed two-level bootstrap, resampling seeds and tasks with replacement, 10,000 resamples, 95% percentile intervals. For comparisons with the matched control the control's value for a (seed, task) cell is its mean over the 20 draws.
- p-values: two-sided, from the same bootstrap (twice the smaller tail proportion, floor 1/10,000). Holm correction within the primary family.
- A comparison is called not distinguishable when its interval includes zero; the interval is always reported.

## Interpretation of the planned comparisons

- If F holds across seeds, it supports an effect of replanning frequency. If success follows the budget per step independently of the replanning period, the comparison supports an effect of search effort.
- If a rule beats its matched control (Holm-corrected) the timing reading changes from null to positive for that rule; if the control beats the rule, the pilot's direction is confirmed with a matched control; otherwise the result is a null with its interval.
- The pilot's post hoc observations about the distribution of replans remain exploratory unless the matched comparisons support them.
