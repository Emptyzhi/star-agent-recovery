# Release validation

Validation was performed without starting a model or benchmark experiment.

- Main-table reconstruction: 63 cohorts, 23,814 task records, all 21 rows match.
- Published Python sources: all parse successfully.
- STAR offline suite: 288 tests, 284 passed and four skipped.
- AgentRewind offline suite on Linux: 409 tests passed.
- Shared baseline transport and history checks: 15 tests passed.
- LATS framing, scoring, and recovery-stage checks: 19 tests passed.
- ToT framing, label, and purchase-scoring checks: 19 tests passed.
- Pinned external dependency acquisition and compatibility reconstruction
  completed successfully. Generated upstream files are excluded from Git.

The retained legacy ToT/LATS suites include older interface and round-setting
expectations; broad discovery is not a passing release gate. The obsolete
test importing the absent LATS candidate-pipeline module has been removed.
The targeted checks above cover framing, scoring, and stage accounting; they
do not establish a fresh model-level rerun or exact historical-code parity.

The publication copy changes resource locations and descriptive comments,
repairs an inherited test indentation error, and aligns one replay-window
test with its existing prefix-plus-new-actions semantics. It does not change
archived outcomes or the recovery algorithms to improve benchmark scores.

Publication checks cover tracked author identifiers, private paths, signatures,
and institution names. Upstream dependency URLs remain in `upstreams.json`.
