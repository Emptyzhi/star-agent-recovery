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

The retained legacy ToT/LATS `tests/` suites include expectations from older
interfaces (including a former two-round ToT setting and a removed candidate
pipeline). Broad discovery of those legacy suites is not a passing release
gate. The targeted checks above exercise the released protocol, scoring, and
stage-accounting boundaries. They do not establish a fresh model-level rerun
or byte-identical reproduction of every historical deployment.

The publication copy changes resource locations and descriptive comments,
repairs an inherited test indentation error, and aligns one replay-window
test with its existing prefix-plus-new-actions semantics. It does not change
archived outcomes or the recovery algorithms to improve benchmark scores.

Before publication, tracked filenames and content were checked for known
author identifiers, personal machine paths, email signatures, and institution
names. External repository URLs in `upstreams.json` identify dependencies;
they are intentionally retained so a reader can obtain the original sources.
