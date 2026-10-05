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
| 2026-10-03 | 5 | 4 | 0 | 0 | 5 | 1 of 4 (#351 amended) | #349, #350, #352 ready as written; #351 amended with the workbook arm; #345 had no PR, fixed on `fix/345-permission-answer-undelivered` | none (first record) | 2 described, 0 requested (no CLI) |
| 2026-10-05 | 6 | 5 | 0 | 1 | 6 | 3 of 5 (#371 and #372 superseded, #373 rebuilt) | #369, #370 merged as written; #373 rebuilt as #377 after its test stopped compiling on `main`; #371 superseded by #374; #372 superseded by #376; #363 had no PR and was fixed as a class in #375 | none | 1 described, 0 requested |

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

## 2026-10-03 — first triage under the skill

41 issues, 2026-09-25 through 2026-10-03; 5 open, all from this batch.

| month | n | no attr | median age | p90 age | <30d | >90d |
|-------|---|---------|-----------|---------|------|------|
| 2026-09 | 36 | 1 | 8 | 11 | 35 | 0 |
| 2026-10 | 5 | 0 | 2 | 2 | 5 | 0 |

**Reading:** PR≤3d is 5 of 5, but only three findings (#344, #345, #346) are
defects in logic the blamed PR added (#331, #336, #333, all merged
2026-10-01); #347 and #348 are older defects blamed on the last PR to touch
the line. The Detail-only column stays at 0: no finding was in a Detail fix
PR. Nothing matched residue, since this is the first record. Still too young
for the age columns to separate backlog from regression.

### Classes

Every finding landed in a different keyword cluster (+1 each to audit, data
loss, ordering, stale state, fail-open, lifecycle), and the sibling hunt
confirmed one class: DuckDB reader errors persisted with the absolute
`files/` path (#347), four arms across `TableLoad::create` and
`WorkbookLoad::load`, three fixed by #351 and the fourth by its amendment.
#345 (allow after the stream disconnected audited `allowed`, 204) joins the
audit-outcome cluster, now 9 findings over four batches with no rule
requested; its fix is on a branch of its own with a `Delivery` type as the
guard.

### Decided

Instances merged as written: #349, #350, #352. #351 amended. #345 fixed on
`fix/345-permission-answer-undelivered`. Two rule descriptions recorded in
`.local/detail-triage-2026-10-03.md` for a machine with the `detail` CLI.
Residue: the terminal's `named_tables` still ends a comma `FROM` list at a
subquery; the workbook arm has no test.


## 2026-10-05

47 issues, 2026-09-25 through 2026-10-05; the batch's 6 were open at triage
and all are closed.

| month | n | no attr | median age | p90 age | <30d | >90d |
|-------|---|---------|-----------|---------|------|------|
| 2026-09 | 36 | 1 | 8 | 11 | 35 | 0 |
| 2026-10 | 11 | 0 | 1 | 2 | 11 | 0 |

**Reading:** PR≤3d is 6 of 6, and all six are defects in logic the blamed PR
added: two in #358 (file drop), three in #359 (ratatui widgets), both merged
the day of the scan, and one in #352, Detail's own reranker fix from the
2026-10-03 batch, which that triage passed as written. New code is producing
the findings; this is not backlog.

### Classes

One confirmed class: a `background_effort` the model rejects passes
`quack doctor` and then fails at use (#363), with the history summarizer as a
second site on the turn path. It joins "config passes, runtime fails" (now 4).
#365 and #366 share a cause (shlex's shell rules applied to a line of file
names) but are two members with different fixes, not a class of three.

### Decided

All six issues closed the same day.

- Merged as written: #369 (#364) and #370 (#365).
- #373 (#368) failed its merge queue checks twice: its test called
  `Message::lines`, which #362 had renamed the same day. Rebuilt on `main`
  as #377.
- #371 (#366) added a hand-written shell tokenizer. Superseded by #374: a
  `FileLine` type refuses a line of names cut short by a `#` word, with shlex
  still the only parser.
- #372 (#367) kept or dropped a call's text by whether the previous call ran
  a tool, so `A.` + tool, `B.` + tool, `C.` answered `B. C.` (reproduced).
  Superseded by #376, which holds a call's text as provisional until its own
  tool runs.
- #363 had no PR. #375 fixed the class at both turn-path sites and added the
  `background_effort` check to `quack doctor`.

A flaky server test surfaced in the merge queue
(`a_waiting_write_is_answered_once_by_its_asker_or_expires`, 404 for 409 on
a slow runner): its answer cases raced its own one-second expiry. It came in
with the 2026-10-03 batch's permission fix and was split in #377.

No rule was requested. The description for the `background_effort` class is
in `.local/detail-triage-2026-10-05.md`. Residue: the reranker's degraded
path in `dispatch` has no whole-turn test; a rejected call's text still
streams as live deltas before rig rejects it.
