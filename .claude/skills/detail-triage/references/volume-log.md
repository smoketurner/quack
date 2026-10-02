# Detail Volume Log

Measured history of Detail's findings, so a later run can tell a trend from a
blip. Add a row to the batch table and append one section per triage pass.
Counted figures come from `scripts/detail-stats.py`; do not hand-edit them. The
judgement columns come from the batch's record in `.local/`.

## Batch table

| column | source | meaning |
|---|---|---|
| issues, fix PRs, dead-code PRs | script | what Detail opened on the detection date |
| Detail PR | script | findings blamed on one of Detail's own PRs |
| PR≤3d | script | findings blamed on any PR merged at most 3 days before detection, ours included |
| not as written | record | PRs that needed changes, were superseded, or were closed after review |
| dispositions | record | what happened to the batch's PRs |
| residue | record | findings that match residue or an open decision in an earlier record |
| rules | record | Detail rules requested for the batch's classes, and whether synced |

| batch | issues | fix PRs | dead-code PRs | Detail PR | PR≤3d | not as written | dispositions | residue | rules |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | 29 | 23 | 0 | 0 | 7 | not recorded | all 23 merged (#252–#274); remaining auth, audit, and token findings closed by #275 | none (first batch) | 0 |
| 2026-09-28 | 6 | 5 | 0 | 1 | 5 | not recorded | all 5 merged (#297–#301); #296 closed by class fix #302 | not recorded | 0 |
| 2026-09-30 | 1 | 1 | 0 | 0 | 1 | not recorded | #315 merged | not recorded | 0 |

These three batches predate this skill and have no `.local/` record, so their
judgement columns are reconstructed from GitHub, not from a review.

**Reading:** the 2026-09-28 batch is the first sign of the loop this skill
exists to break. Five of its six findings were in PRs merged in the previous
three days: #296 in Detail's own #262 (the fix for #234), and #291 in #275, the
human class fix that closed the rest of the 2026-09-25 batch.

## 2026-10-02 — baseline

36 issues, 2026-09-25 through 2026-09-30, all closed.

| month | n | no attr | median age | p90 age | <30d | >90d |
|-------|---|---------|-----------|---------|------|------|
| 2026-09 | 36 | 1 | 8 | 11 | 35 | 0 |

Ages are in days between the "Introduced in" attribution and the detection
date. "no attr" counts issues whose body carries no attribution line.

**Reading:** the repository is about three weeks old, so every attributed
finding is under 30 days and the age columns cannot yet separate backlog from
fresh regressions. They become informative after the first month.

### Classes (keyword clustering over titles — directional, not rigorous)

| class | total | 09 |
|---|---|---|
| missing audit event | 8 | 8 |
| silent data loss | 8 | 8 |
| session/history ordering | 8 | 8 |
| stale state / embeddings | 7 | 7 |
| ontology/graph consistency | 7 | 7 |
| fail-open / misreported outcome | 6 | 6 |
| concurrency/race | 5 | 5 |
| job/writer lifecycle | 4 | 4 |
| incomplete revocation | 3 | 3 |
| string canonicalization | 3 | 3 |
| config passes, runtime fails | 3 | 3 |
| authz gap | 2 | 2 |

Missing audit rows are the clearest recurring class: #223, #231, #243, #244,
#245, and #291 across two batches, plus the audit half of #293. No Detail rule
has been requested for it yet.
