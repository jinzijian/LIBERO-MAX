# Cosmos Physical Pilot Calibration

This note records calibration evidence for
`cosmos_physical_pilot_v0.1.json`. It is a pilot record, not a benchmark-wide
result.

## Intervention mechanics

All five cases were applied to the real `libero_object/task_0`, initial state
0, in MuJoCo without loading the policy. Every case produced a non-zero primary
camera change and no simulator error:

| Case | Raw-pixel MAD | Setup events |
| --- | ---: | ---: |
| light off, scale 0.05 | 70.0638 | 0 |
| light on, 0.05 then x20 | 70.0638 | 1 |
| target translation, +6 cm y | 0.8941 | 0 |
| target translation, +12 cm y | 0.8630 | 0 |
| five-distractor burst | 2.4171 | 5 |

The distractor burst hides five task-native non-target objects off-world in
both paired arms, then inserts all five only in the intervention arm.

## Proximity threshold

The successful Cosmos control trajectory from the initial paired smoke was
replayed without model inference. The end-effector first crossed candidate
target-distance thresholds at:

| Threshold | First policy step | Actual distance |
| --- | ---: | ---: |
| 24 cm | 14 | 23.86 cm |
| 21 cm | 21 | 20.98 cm |
| 18 cm | 30 | 17.40 cm |
| 15 cm | 34 | 14.46 cm |
| 12 cm | 37 | 11.92 cm |

The pilot uses 18 cm. On this trajectory the physical change occurs at policy
step 30 and the next 16-action-chunk policy query occurs at step 32, leaving a
two-step open-loop exposure. The replay still reaches the original goal and
has a minimum target distance of 1.33 cm.

This single trajectory establishes that the trigger is reachable and occurs
during approach. It does not establish that 18 cm is the final threshold for
all tasks; multi-task calibration must report trigger coverage before freezing
the v1 manifest.
