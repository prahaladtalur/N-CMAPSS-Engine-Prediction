# Corrected N-CMAPSS maintenance-window warning evaluation

The former 16-channel setup was retired because it included Fc and hs, which are auxiliary N-CMAPSS fields. This result bundle uses only the 14 measured physical channels, separately named elapsed-cycle and operating-condition inputs, and no auxiliary health-state field.

## Primary comparison

| Input | AUROC | Precision | Recall | Pre-window false-alert rate |
| --- | ---: | ---: | ---: | ---: |
| Elapsed cycle only | 0.923 | 0.813 | 0.563 | 0.050 |
| Cycle plus operating conditions | 0.935 | 0.821 | 0.554 | 0.046 |
| Measured telemetry only | 0.939 | 0.863 | 0.771 | 0.047 |
| All observable inputs | 0.966 | 0.857 | 0.793 | 0.051 |

The primary comparison is all observable inputs minus cycle plus operating conditions.
The AUROC difference is 0.031 [0.023, 0.039].
The recall difference is 0.239 [0.183, 0.296].

## Fixed 0.5 threshold check

All-observable precision=0.791, recall=0.872, and pre-window false-alert rate=0.089.
Cycle-plus-operating precision=0.700, recall=0.859, and pre-window false-alert rate=0.141.

This is a simulation result only. It does not establish a maintenance schedule, a safe operational horizon, or fielded-aircraft performance.
