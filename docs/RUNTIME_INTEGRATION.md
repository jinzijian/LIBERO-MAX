# Runtime Integration Contract

LIBERO-MAX separates **when** a change occurs from **how** a simulator applies
it. This keeps trigger semantics testable without importing LIBERO or MuJoCo.

## Episode loop

For each control/intervention pair, restore the exact same initial simulator
state and policy state. In the intervention arm, call the runtime immediately
before every policy query:

```python
runtime.reset(original_instruction)
observation = env.set_init_state(init_state)

for step in range(max_steps):
    event = runtime.maybe_apply(
        TriggerContext(step=step, max_steps=max_steps, events=detected_events)
    )
    if event is not None:
        observation = backend.refresh_observation()
        trace.write(event)

    action = policy(observation, runtime.current_instruction)
    observation, reward, done, info = env.step(action)
```

Applying the change before the policy query guarantees that the first
post-change action is conditioned on the changed observation or instruction.

## Event-backed triggers

The evaluator converts task-progress detectors into these canonical event keys:

| Trigger | Required event key |
| --- | --- |
| `before_grasp: mug` | `pregrasp:mug` |
| `after_grasp: mug` | `grasp:mug` |
| `after_subgoal: open_drawer` | `subgoal:open_drawer` |
| `on_region_entry: transfer_corridor` | `region:transfer_corridor` |

`fixed_step` and `progress_fraction` are derived directly from the runtime
step. Fixed-step triggers are intended for deterministic diagnostics; benchmark
scenarios should prefer semantically matched progress events.

## Exactly-once guarantee

`InterventionRuntime` marks a change as applied only after the backend succeeds.
Subsequent calls return no event. Every successful event records:

- scenario ID and seed;
- simulator step;
- trigger and change payload;
- instruction before and after the change;
- backend-reported before/after state;
- expected response mode.

## Model adapter contract

A shard adapter is a Python entry point with this common interface:

```text
runner.py MANIFEST --output-root ROOT --shard-index I --num-shards N --resume [model arguments]
```

It must load cases with `cases[I::N]`, keep one policy instance resident, write
one terminal `DONE` or `FAILED` marker per case, and preserve Base actions
exactly before the event. Model repositories and weights remain external and
are supplied as explicit arguments.

`scripts/run_dynamic_benchmark.py` is model-agnostic. It splits cases by task
suite, uses all GPUs visible to the process unless `--gpus` is supplied, and
dynamically assigns work units to avoid idle devices near the end of a run.
The scheduler changes resource allocation only; it does not change event
timing, policy cadence, actions, or scoring.

## Reference evaluation adapters

The public adapters below exercise the same frozen manifest and paired replay
contract used for the reported evaluations.

| Model | Family | Multi-GPU launcher |
| --- | --- | --- |
| pi0.5 | VLA | `scripts/run_openpi_persistent_benchmark.sh` |
| OpenVLA-OFT | VLA | `scripts/run_openvla_oft_persistent_benchmark.py` |
| X-VLA | VLA | `scripts/run_xvla_persistent_shard.py` |
| VLA-JEPA | VLA + WM | `scripts/run_vlajepa_persistent_benchmark.py` |
| Cosmos-Policy | WAM | `scripts/run_cosmos_persistent_benchmark.py` |
| Fast-WAM | WAM | `scripts/run_fastwam_persistent_benchmark.py` |

Worker output is written per logical shard and can be resumed without
re-evaluating completed cases. A publishable run must cover all 8,000 case IDs,
contain a terminal outcome for both arms, record the checkpoint and native
query cadence, and pass the exact replay checks before aggregation.

When an installed `libero` wheel shadows a pinned LIBERO-Plus or LIBERO-PRO
source checkout, prepend `scripts/libero_source_overlay` to `PYTHONPATH` and set
`LIBERO_SOURCE_PACKAGE_ROOT` to that checkout's outer `libero` directory. The
overlay changes import resolution only. It does not patch the simulator,
benchmark cases, or upstream source code.

## Current MuJoCo operations

The reference `LiberoMujocoBackend` implements:

- `shift_camera` through MuJoCo camera position/quaternion updates;
- `move_object` for free-joint objects or fixed fixtures;
- `insert_obstacle` when the obstacle is preloaded in the scene;
- `remove_object` by moving the named entity to an off-world position;
- `set_lighting` by scaling MuJoCo light parameters.

Intent changes (`replace_instruction` and `cancel_instruction`) are handled by
the runtime because they modify the next policy query rather than simulator
physics.

## Cosmos Policy integration

`CosmosInterventionEnv` applies physical interventions after an environment
step and only at a Cosmos action-chunk boundary. The observation returned to
the upstream evaluator is regenerated after the change, and the action queue
is empty at that boundary, so the next action chunk is queried from the
post-change observation.

The integration records an SHA-256 digest of the serialized LIBERO initial
state in each arm. A paired summary is valid only when control and intervention
digests, task descriptions, suite names, task indices, and policy seeds match.
The first reference smoke uses standard `libero_object` task 0, initial state 0,
seed 195, and a camera shift after the first 16-action chunk.

This wrapper currently supports physical changes only. Intent changes require
changing the instruction passed into each Cosmos policy query and are rejected
by the launcher until that query-level hook is implemented.
