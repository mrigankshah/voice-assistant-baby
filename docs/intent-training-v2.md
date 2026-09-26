# Local intent classifier: first trained comparison

Run date: September 26, 2026. Windows CPU, two PyTorch threads. No Pi measurements yet.
The live assistant still uses the existing router. No tools ran during evaluation.

## Artifacts

- Frozen encoder baseline: `models/intent-minilm-v2/`
- SetFit candidate: `models/intent-setfit-v2/`
- Original downloaded encoder retained: `models/intent-minilm/encoder/`

Both new artifacts use the same 248 training examples, normalized embeddings, and
balanced logistic regression head. SetFit additionally fine-tunes the MiniLM encoder
for 150 steps with seed 42. Training took about 226 seconds. The saved encoder and
JSON head are sufficient for inference; SetFit is only needed for training.

Dataset SHA256: `3a2a32793a3e3585f260044d3b81aa8de836b6b04bb6e398d55ee10aae7b781b`.
These are manually authored scenarios, not sampled real usage. The six labels are
chat, weather, clock, read_setting, write_setting, and unsupported_action.

## Results

Counts below are distinct requests. Each raw validation/test benchmark repeated
requests three times, with the same classification results on each repeat.
Latency is warmed single-request classification, excluding Python/ML library
startup, STT, slot extraction, and LLM answer generation.

| Model | Validation | Held-out test | Test median | Test P95 |
| --- | --- | --- | --- | --- |
| Frozen MiniLM | 48/59 (81.4%) | 45/53 (84.9%) | 26.8 ms | 29.4 ms |
| SetFit MiniLM | 50/59 (84.7%) | 48/53 (90.6%) | 31.1 ms | 49.8 ms |

SetFit test breakdown:

- Standalone starter scenarios: 35/36 correct. This group retains a few original
  hypothetical/negated cases; it is not a pure estimate of everyday traffic.
- Contextual scenarios: 5/8 correct.
- Unsupported action families excluded from training (music, smart home, shopping):
  8/9 correct. This small sample does not establish universal unknown-action detection.

The five raw test failures were:

| Request | Expected | Predicted |
| --- | --- | --- |
| Don't switch my units to Celsius | chat | write_setting |
| Are you having a good day? (after weather) | chat | weather |
| And the following day? (after weather) | weather | clock |
| Set the thermostat to twenty degrees | unsupported_action | write_setting |
| Play some relaxing music (after weather) | unsupported_action | chat |

## Uncertainty experiment

A threshold of 0.6 was chosen for a diagnostic test from the validation sweep,
before inspecting held-out predictions. It is not a recommended deployment threshold.

- Validation: accepted 43/59; 41/43 accepted predictions correct.
- Held-out test: accepted 37/53; 35/37 accepted predictions correct.
- Separate challenge set: 7/14 expected outcomes matched, with 10 rejections.

At that threshold, the negated settings command and thermostat request still
route to write_setting. Scores are not calibrated probabilities. Rejection alone
does not fix the semantic mistakes, and misrouted requests would still require
the existing workflow's argument/authorization checks before any execution.

## Next work

1. Use this saved SetFit candidate for an offline Pi timing/memory test while STT
   and Ollama are loaded. Keep it resident instead of loading for every utterance.
2. Improve contextual training beyond the current previous-weather-task examples.
   Include topic switches and compact state from every supported workflow.
3. Add realistic distinctions between changing assistant preferences and controlling
   an external device. Keep rare negations a separate robustness focus.
4. Create a fresh held-out set for the next training iteration. The failures above
   are now inspected diagnostics, so tuning to them cannot establish generalization.
5. Integrate only after testing slot extraction, missing-field dialogue, cancellation,
   and the registry/authorization checks end to end. This classifier selects a route;
   it does not extract arguments or execute actions.

## Raw reports (local, ignored by Git)

- Frozen validation: `eval-results/intent-20260926-151112-137060/`
- SetFit validation: `eval-results/intent-20260926-151119-447243/`
- Frozen test: `eval-results/intent-20260926-151235-557643/`
- SetFit test: `eval-results/intent-20260926-151241-334182/`
- SetFit test at 0.6: `eval-results/intent-20260926-151243-357588/`
- SetFit challenge at 0.6: `eval-results/intent-20260926-151244-168130/`

Each directory contains `report.md` and `results.json`. Dataset/artifact hashes,
package versions, per-request scores, and timings are recorded. Training and
evaluation completed with exit code 0. This Windows Python 3.12 environment also
printed a multiprocess ResourceTracker cleanup exception at process exit; model
files were saved and successfully reloaded for evaluation.

The repository unit suite passed 79 tests. Those tests validate code and dataset
checks; they are separate from measured model accuracy above.
