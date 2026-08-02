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
