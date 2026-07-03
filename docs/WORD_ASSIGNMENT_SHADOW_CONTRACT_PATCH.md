# Word assignment contract-compliant implementation

This patch starts from the clean rollback branch and deliberately does **not** change live runtime behavior yet.

It adds a contract-compliant word-assignment shadow planner that preserves the original runtime contract:

- committed tracks remain absolute-time event schedules;
- committed events remain append-only and immutable;
- the performer/consumer contract is untouched;
- word assignment only produces diagnostic absolute-time candidate events.

New per-case outputs:

- `word_assignment_shadow.csv`
- `word_assignment_contract_summary.csv`

The shadow planner computes absolute event centers from observed speech-region word spans where mapping is safe, otherwise from text-duration projection. It compares those candidate centers to the existing committed track and logs contract safety fields before any runtime promotion.

Use this to inspect whether the word-assignment planner is temporally safe before replacing the old scheduler.
