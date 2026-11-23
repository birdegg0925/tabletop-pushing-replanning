# Stage 3 analysis plan

Recorded on 18 November 2026, before the Stage 3 reference simulations.

## Why Stage 3 exists

The Stage 2 results showed that the Stage 2 random-timing control matches each rule on replans per executed step pooled over the episode, but not on when in the episode the replans fall. With state input, in steps 0 to 9 the rules replan more often than their matched controls (for example the execution-error rule at the 50th percentile: 0.52 against 0.40 replans per step), and less often later. Most successes occur in those steps. The Stage 2 advantages of some rules over their controls (best coverage at the pre-specified threshold; four arms of the threshold sweep) are therefore not separated from denser replanning early in the episode. This was computed from the Stage 2 result files before this plan was written; it is the reason for the plan, and the Stage 2 numbers are not re-used as Stage 3 outcomes.

The full Stage 2 results, the early-window replanning rates and the block's coasting distance across a grid of damping values were available when this plan was recorded. Coasting distance was measured without a controller. At that point, no Stage 3 episodes had been run.

## Part A. Schedule-shuffled control (primary)

**Construction.** For a model and a rule arm, Stage 2 recorded the rule's replanning schedule on each of the 64 test tasks as a string of 35 bits (replan or reuse at each step of the full 35-step episode). The shuffled control for task i and draw k runs task i with the schedule that the same rule, with the same model, produced on another task π_k(i), where π_k is a random permutation of the 64 tasks with no fixed point (generated from a seed made of the model's training seed, the index of the rule arm and k). The controller replans exactly at the steps where the borrowed schedule has a 1. Planner, candidate count, search rounds and planner random numbers are those of the rule's own episode on task i. Twenty draws per task, k = 1 to 20.

**What it holds fixed.** Within each draw every task's schedule is used exactly once, so over the 64 tasks the control uses the same collection of schedules as the rule: the same number of replans at every step of the episode, the same runs and gaps, the same total. What it removes is the link between a schedule and the state of the episode it is applied to. A borrowed schedule never leaves a cached plan in use for more than six steps, because the rule's own schedules never do.

**Check of the construction (before outcomes are read).** For every model and arm: (1) replans per step over the full 35 steps are identical for rule and control (check equality in the episode files); (2) replans per step in steps 0 to 9 over all episodes are identical (exact). Realised replans per executed step up to first success, and the rate in steps 0 to 9 among episodes still running, depend on outcomes and are reported for both without a tolerance.

**Models and arms.** State input, ten seeds; image input, five seeds; all 13 rule arms (three scores at four thresholds, and the geometry rule); damping 0; the 64 test tasks of Stage 2 (task seed 83). Episodes run for 35 steps.

**Outcomes.** Success (coverage above 0.95 at any step within 35) and best coverage over the 35 steps, as in Stage 2.

**Comparisons.**

- Primary family (state input; eight comparisons; Holm correction across the eight): each of the four rules at its pre-specified setting (execution error, ensemble disagreement and predicted near-contact at the 75th percentile; the geometry rule) minus its shuffled control, on successes and on best coverage. The control's value for a (seed, task) cell is its mean over the 20 draws.
- Secondary (reported with intervals, uncorrected, labelled as such; a Holm correction across the 13 arms is shown for reference):
  - every rule arm of the sweep minus its shuffled control, on both outcomes;
  - shuffled control minus the Stage 2 rate-matched control, for every arm, on both outcomes. Since rule − rate-matched control = (rule − shuffled control) + (shuffled control − rate-matched control), this splits the Stage 2 difference into a part due to the state the replans respond to and a part due to the schedule's shape alone;
  - the same comparisons for image input.

**Inference.** As in Stage 2: unit (seed, task); crossed two-level bootstrap over seeds and tasks, 10,000 resamples, 95% percentile intervals; two-sided p-values from the same bootstrap; "not distinguishable" when the interval includes zero.

**Interpretation of the comparisons.**

- A rule's timing is credited only where rule minus shuffled control excludes zero after the Holm correction in the primary family.
- Where rule minus shuffled control includes zero and shuffled minus rate-matched control excludes zero, the Stage 2 advantage of that rule is attributed to the shape of its schedule (how its replans are spread over the episode), not to the signal.
- Where both include zero, the Stage 2 advantage is reported as not resolved by this decomposition.
- The Stage 2 control matched the overall replanning rate. Its replans could fall at different points within an episode.

## Part B. Milder changes of damping (secondary)

**Reason.** At the smallest Stage 2 value (0.02), the block coasted 38.0 px after release compared with 4.6 px in training. The simulator-scored planner finished only 32 of 64 tasks at that setting. A grid measured before this plan identified milder shifts. Coasting distance was 4.6 px at 0, 6.2 at 1e-12, 8.0 at 1e-9, 11.5 at 1e-6, 13.7 at 1e-5, 16.9 at 1e-4, 19.1 at 3e-4, 22.3 at 1e-3, 24.6 at 2e-3, 26.3 at 3e-3, 28.7 at 5e-3, 32.7 at 1e-2, 38.0 at 2e-2.

**Values.** Fixed by rule from that grid: the grid values whose coasting distances are closest to one quarter, one half and three quarters of the way from 4.6 px to 38.0 px (13.0, 21.3, 29.7 px), which are 1e-5, 1e-3 and 5e-3; and 1e-9 as the mildest value clearly different from the training condition (8.0 px). Four values.

**Runs.** Simulator-scored planner, every step, 64 test tasks, at each value (episodes stop at first success, as in Stage 2). Learned model, state input, ten seeds: every step, every four steps and the execution-error rule at the 75th percentile at each value (the three schedules Stage 2 ran under unseen damping). Goals are generated under each value, as before.

**Reading.** The shift is called graded if the simulator-scored planner succeeds on more than 32 of 64 tasks at one or more of the four values. The learned controller's successes are then reported at each value beside the simulator-scored planner's, with the difference between every step and every four steps and its interval, and the trigger rate of the execution-error rule. This comparison is descriptive.

## Part C. Level of the execution-error score when nothing is happening (secondary)

**Reason.** The Stage 2 thresholds were expressed in pixels. This check measures the score floor while the block is nearly stationary.

**Measurement.** For each model, on its 48 calibration tasks with the every-six-steps schedule (the run that sets the thresholds), record at every step the execution-error score and the true displacement of the block over the preceding step. The floor is the median of the score over steps in which the block moved less than 0.5 px; its 90th percentile is also reported. Each of the model's four thresholds is then expressed as a multiple of the median floor. State input and image input. Descriptive; no test.
