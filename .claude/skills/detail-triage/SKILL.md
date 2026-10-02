---
name: detail-triage
description: Triage a batch of Detail bug issues and fix PRs — measure the trend, classify each finding as an instance or a class, review each fix for defects in the fix itself, and decide what merges versus what gets fixed as a class with a guardrail. Use when asked to "triage the Detail batch", "Detail opened a batch", "why is Detail volume rising", "classify Detail findings", or when a new batch of `[Detail Bug]` issues appears.
---

# Triage a Detail Batch

Detail scans the whole repository on each pass. It files one issue per finding,
usually with a paired fix PR. Merging each PR on its own closes the instance and
leaves the pattern in place for the next pass to find somewhere else. This skill
breaks that loop: measure what is happening, separate instances from classes,
and make every class fix carry something that keeps the class from coming back.

Work through the steps in order. Step 2 gates everything after it: nothing
merges before the classification table exists.

## Step 0: Know what you may not edit

**Detail rules are Detail's.** Detail generates every rule; we request rules and
sync the generated files into the repository. We never write a rule, and we
never edit a synced one: Detail scans with its own copy, so a local edit never
reaches the scanner, and the next sync overwrites it.

- `detail rules pull` writes into `.claude/skills/detail-rules/`, and
  `detail skill rules` installs `.claude/skills/detail-create-rules/`. Both are
  generated. Never edit files in either directory by hand.
- A rule that is wrong or stale gets a new request through the CLI (step 5),
  never a local fix.

Everything under `.claude/skills/detail-triage/` is repo-owned and yours to
maintain.

## Step 1: Measure the batch and the trend

```bash
python3 .claude/skills/detail-triage/scripts/detail-stats.py
python3 .claude/skills/detail-triage/scripts/detail-stats.py --json   # for scripting
```

The script reads every Detail-authored issue from the GitHub REST API. It takes
the "Introduced in … on DATE" attribution from each body and reports volume, bug
age, and keyword-clustered classes per detection month. The attribution names a
PR (`[#275](…)`) or, for code pushed to `main` before it went through PRs, a bare
commit SHA; a SHA gives the age but no PR. The script needs no `gh` CLI: the
token comes from `GH_TOKEN` or `GITHUB_TOKEN`, falling back to `gh auth token`,
and the repository comes from `GH_REPO` or the `origin` remote.

Read three things together:

- **median / p90 age**: how far back into history this pass reached.
- **`<30d`**: findings against code merged in the last month.
- **`Detail PR` and `PR<=3d`** (per-batch table): findings blamed on one of
  Detail's own fix PRs, and findings blamed on any PR merged at most three days
  before detection, whoever wrote it. The same table counts Detail's fix PRs and
  its Dead Code PRs, which file no issue.

A climbing median with a flat `<30d` means Detail is working through backlog.
The volume is high for a reason no process change fixes, and the response is
throughput. A climbing `<30d` means new code produces findings as fast as old
code is cleaned up, and the response is a guardrail. Say which of the two you see
before proposing any remedy. quack is young, so nearly every finding is under 30
days old; `<30d` only separates the two populations once the history is longer
than a month.

`PR<=3d` decides how this batch gets merged. When it is high, merging fix PRs
as-is feeds the next scan, and the loop breaks only by making class decisions
*before* merging. Part of the signal is mechanical: the more fix PRs merge, the
more often a fix PR is the last commit to touch a line. Confirm a spike by
checking that the finding is a defect in the logic the fix *added*, not merely in
a file it touched. quack's first two batches show both sources:

- **Detail's own fix.** On 2026-09-25, #234 reported job cleanup callbacks lost
  on history eviction, and Detail's #262 fixed it. On 2026-09-28, #296 found
  that `when_ended` cleanup could still be dropped under concurrent job finishes,
  a defect in the logic #262 added. The class fix #302 ended a job, delivered its
  waiters, and evicted under one lock.
- **A human class fix.** #275 closed the remaining auth, audit, and token
  findings from 2026-09-25. Three days later #291 found that the OIDC callback in
  #275 committed the subject token before writing its `Login/Allowed` audit row.
  The Detail-only column read 1 for that batch; `PR<=3d` read 5 of 6.

The class table is **keyword clustering over titles: directional, not
rigorous**. Titles overlap classes. Use it to spot recurrence worth
investigating; never quote its counts as fact.

Compare against `references/volume-log.md`, which holds the measured history;
its batch table is the per-batch series to extend.

## Step 2: Classify every open finding

For each open issue in the batch, establish three things and put them in one
table:

| column | what it means |
|---|---|
| class | which recurring pattern it belongs to, or "isolated" |
| live? | is the defect still present in current `main`? |
| siblings | other call sites sharing the pattern, found by reading the code |

**Verify "live?" against the tree, never against the issue body.** Detail writes
a batch from a snapshot, and that snapshot can predate a merge from the same day.
A quoted line can be gone while the defect survives a few lines away, or the
reverse. Read the current file.

**Match against stated residue before hunting.** Read the residue and
open-decision sections of the last few records in `.local/`. A finding that
matches something a review already accepted is a residue recurrence, not a
surprise. Count it in the batch table. The lever for a residue recurrence is
fixing residue in the PR that left it, not a new guardrail.

Finding the siblings is the work of this step; it turns a list of instances into
a class. Prefer `ast-grep` or the rust-analyzer LSP over ripgrep here: the
question is structural ("every handler that propagates an error with `?` before
it writes its audit row"), not textual. Detail's own bodies often name the
sibling that already does it right: #244 found that the REST `search` handler
audited `Outcome::Allowed` before propagating, while the `sql` handler beside it
derived the outcome from the result. Every handler that audits is in the class
until read.

**The sibling hunt disqualifies classes as often as it confirms them, and that
is a result worth having.** Exclude test-only sites, and judge each sibling's
consequence rather than counting matches. A sibling that only affects what the
UI displays, or that fails in the safe direction behind an authoritative guard,
is not a member. A class of three where two members are harmless is not a class.
Say so, and merge the instance.

Issues Detail filed **without** a paired fix PR belong in this table too. Detail
usually judged them too structural to auto-fix, which makes them the strongest
class candidates in the batch, not the weakest.

Check the class against quack's critical paths in
`.claude/rules/continuous-improvement.md` and the gates in
`.claude/rules/code-standards.md`. A finding on one of them (audit rows for every
workspace-touching request, the classification boundary, workspace connection
confinement, session recording order, citation validation, the embedding
profile) is a class until the sibling hunt proves otherwise.

### Dead Code PRs

Detail also opens Dead Code PRs (branch `detail/dead-code/…`) that file no
issue. CI builds every target, so the compiler proves most removals: a function,
impl, or constant that still had a caller would not build. Review the removals
the compiler cannot see:

- **A field on a stored or interchange shape.** A workspace file written by an
  older quack must still open, and an exported bundle must still import. Check
  the JSON kept inside the workspace (ontology snapshots in
  `_quack_ontology_*`, message metadata in `_quack_messages`), the OKF bundle,
  `AgentResponse::to_json` (the REST, MCP, and print-mode contract), and the
  ontology interchange types, which use `deny_unknown_fields`: a field removed
  there makes every older export fail to import. A field no code reads can still
  be a contract.
- **A column or table in a migration.** A shipped `control.db` migration is
  never edited (`docs/migrations.md`); removing what it created takes a new one.
- **Config keys.** A setting that stops being read becomes an unknown key that
  `quack config` and `quack doctor` report on every existing install. Remove it
  from the docs and `deploy/config.toml` in the same PR.
- **Unreachable branches kept on purpose**, such as a defensive arm that keeps
  a degraded path from panicking. Read why it is there before accepting the
  deletion.

## Step 3: Decide instance versus class

**A Detail fix PR is a reviewed draft, never a merge candidate.** Green CI, a
detailed PR body, and a full test suite make a PR look finished. Step 6 exists
because that appearance is the same whether the fix is right or not.

The policy is not a claim that Detail writes bad fixes. Measure it with
`detail-stats.py --fix-defect-rate YYYY-MM-DD` before quoting a rate; quack has
too few merged PRs past the window for the number to mean anything yet. The point
is that post-merge blame understates the defect rate, because the scanner does
not re-find everything it introduces. Review closes that gap, whoever wrote the
fix.

Three or more open instances of one pattern makes this decision mandatory. For
each class pick one and record which:

- **Instance**: genuinely isolated. Merge Detail's PR once it has been through
  step 6 and is green. No surrounding refactor.
- **Class**: one change that fixes every site found in step 2, per
  `development-discipline.md` rule 6, "fix the class, not the instance". Close
  Detail's individual PRs as superseded rather than merging them, and say so in
  each.

**Never merge the instance PR for a finding that belongs to a class.** Merging
it closes the instance and erases the evidence that the class exists, so the
next pass finds the same pattern at another call site. #234 then #296 is that
loop in quack's own history.

**Settle contradictory findings together.** A batch can hold findings that pull
in opposite directions, or a fix that reverses a decision already on record.
Resolve the governing text once (the RFC, the design doc section, the memory, the
PR that made the decision), then dispose of the whole set against it.

A class fix that lands without a guardrail will regress, so step 4 is part of
the same PR, not a follow-up.

## Step 4: Attach a guardrail to every class fix

Pick the strongest mechanism that fits, and be honest about what it does not
cover.

1. **A type whose invalid state cannot be constructed.** The strongest option:
   the wrong thing stops compiling. A handler holding `Access` is already
   authorized and the denial already audited, because `Access::resolve` is the
   only way to get one. `Vector` is checked against the profile's `Dimension`.
   Best fit whenever the class is "acted on a value that was never checked".
2. **A clippy `disallowed_methods` entry** in `.clippy.toml`, as exists for
   `EmbeddingModel::embed_texts` and rig's model `call` (embed through
   `Embedder`). Cheap and real (`-D warnings` makes it a CI failure), but it
   matches *a named function*, nothing more.
3. **A behavioral test** asserting the invariant across every site. Never a test
   that scans source text.

**A guardrail can under-cover its class.** A lint on a constructor cannot see
how the constructed value is used afterward, and a type that guards one axis
says nothing about the next. When you pick a mechanism, write down the part of
the class it does not cover, and either add a second mechanism or state the
residue in the PR.

Verify the guardrail the way the project verifies tests: break the code, confirm
CI catches it, then fix it. A guardrail never observed failing is not known to
work.

**A class fix produces the next batch's findings.** Rule 6 of
`development-discipline.md` applies to the mechanism the fix introduces, not
only the one the issue reported. #275 and #262 above are both this. Before
merging a class fix, hunt siblings of the new type or guard: every consumer of
the old behavior it replaces, and every caller it did not touch.

Look past the one interface the issue names. quack has five clients of one core
(CLI and print mode, terminal, web UI, REST, MCP), so a fix in one handler
usually has a sibling in another. When a fix changes who may do something, check
every askama template under `crates/quack/templates/` that renders the action.
When it changes what a stored record means, check every reader of the record:
the web UI, the audit export, `AgentResponse::to_json`, `docs/`, and the
"Interfaces today" section of `CLAUDE.md`.

## Step 5: Request a Detail rule for each confirmed class

This step reduces future volume. A class the scanner knows about is reported as a
rule violation on the way in, rather than found instance by instance for months.

**Request the rule; never write one.** `detail rules create` submits a request
and *Detail* generates the rule text. Writing a rule file, or editing one under
`.claude/skills/detail-rules/`, is forbidden (step 0): Detail would not know
about it, and the next sync overwrites it.

`detail skill rules` installs the `detail-create-rules` skill if it is missing.
The evidence is already in the issues: `detail-stats.py --json` emits an
`open_bug_ids` map from issue number to the `bug_<uuid>` in its body. The class
fix's commits are evidence too.

```bash
detail rules create --description "<the invariant, stated as a rule>" \
                    --bug-ids <bug_id1,bug_id2,...> \
                    --commit-shas <class fix sha,...>
```

Then poll with `detail rules requests show <rcr_...>`, review each result with
`detail rules show <rule_id>`, pull with `detail rules pull <rule_id>`, and
commit the synced files unchanged as `chore(detail): …`.

**Read the generated rule against the merged tree before pulling it, and read
its correct-pattern section, not only its detection section.** Detail generates
from the bug-report snapshot, which predates the batch's own fixes, so a rule can
hold up as "already correct" the code the batch just replaced. That is worse than
no rule: it tells the next reviewer the defect is the model. The remedy is a
fresh `detail rules create` whose description names the stale rule and says what
it got wrong. The CLI has `create`, `propose`, `requests`, `list`, `show`, and
`pull`; there is no refine verb.

A good rule states an invariant, not an incident. "A handler must write its
audit row with the outcome derived from the result, before propagating the
error" is a rule; "issue #244 was fixed wrong" is not.

When a rule for the class **already exists and did not catch it**, say so in the
description and ask for refinement rather than a second overlapping rule.
More overlapping coverage is rarely the lever.

A synced rule gates nothing by itself: the `detail-rules` skill runs only when
someone asks. Wiring it into CI, a hook, or a Makefile target is a separate
decision to raise with the user, not assume.

## Step 6: Review each fix for defects in the fix itself

This step earns the triage. Do not treat it as a merge checklist.

### Scope the reading first

Most of a Detail PR is tests. Split production from test additions so the batch
is tractable:

```bash
python3 .claude/skills/detail-triage/scripts/detail-stats.py --pr-diff-sizes 297-301
```

Read the production hunks. Read tests only to check that the assertion pins the
behavior the issue describes. Before reporting a test as missing, search the
PR's test files for it: a diff filtered to production paths hides them.

The split is by file path, and most quack modules carry an inline
`#[cfg(test)] mod tests`. A PR reporting zero test lines has inline tests, not no
tests; treat its production figure as an upper bound.

### The failure-mode checklist

Interrogate what the fix introduces, not just whether it addresses the report:

- **A new lock, channel, or callback**: does every path that ends the thing
  deliver its waiters under the same lock, or can a concurrent finish drop one?
  (#296 in the cleanup #262 added.)
- **A new retry**: does each attempt re-read state, or capture once outside the
  loop?
- **A new cap or bound**: what happens at exactly zero, and at equality?
- **A new guard**: does it cover every call site and every interface, or only
  the reported one? Search the codebase for the pattern; do not trust the diff's
  coverage.
- **A new branch or match arm**: does anything *upstream* stop it being reached?
  A new arm inherits every early return above it. Read the enclosing function
  from its top, not from the diff hunk.
- **A new write next to an audit row**: is the audit row written before the
  change commits, or in the same transaction? A change that commits and then
  fails to audit stands unaudited (#291). Control-plane changes take their
  `AuditEntry` in the same transaction.
- **A new write to a field another path also writes**: do the two paths share an
  invariant, and does the new one carry it?
- **A new database step**: does it go through the writer (`Writer::run`,
  `with_db`) or a read-only reader, and does it hold any lock across an await?
- **A new normalizer or parser**: does it alter input it should leave alone?
- **A new error path**: does it swallow, and does that match how the adjacent
  code treats the same error? quack degrades with a warning rather than
  aborting; a fix that adds a panic path is wrong under the lints anyway.
- **A new requirement on configuration or stored data**: does every existing
  config file, workspace file, and export still load? Does `quack doctor` check
  what the runtime now requires (#314: a model doctor passed failed its first
  turn)?
- **A changed serde shape**: the compatibility checks under Dead Code PRs
  (step 2) apply to any field a fix removes, renames, or tightens.
- **Anything workspace-revealing**: it stays in the workspace DuckDB file, never
  `control.db`.

### Check the fix against the class it fixes

When the bug is a parsing, normalization, or canonicalization defect, the fix is
written in the same idiom that produced it and tends to inherit the same blind
spot. Test the fix against the spec's own examples and against neighboring
inputs in the same class: non-ASCII text, escaped separators, empty values.
Case-folding is not length-preserving (`İ` lowercases to two characters), so an
offset found in a lowercased string cannot be mapped back into the original by
counting.

A test whose input is derived from the code's own output cannot find a
disagreement with the real producer. When a function exists to accept external
input (a PDF, an XLSX sheet, an issuer's token, a provider's response), require
at least one fixture captured from the external producer.

### Reproduce, do not argue

For a suspected defect in a pure function, extract the function body into a
scratch file and run it. It takes a minute and turns "this looks wrong" into a
confirmed blocker with output to paste into the review:

```bash
rustc -O -o "$TMPDIR/t" "$TMPDIR/t.rs" && "$TMPDIR/t"   # the PR's function body, verbatim
```

A review comment saying "I think this mishandles Unicode" invites debate. One
showing the wrong output does not.

### Then the hygiene checks

- Confirm a cited RFC or design doc section says what the PR body claims: open
  it and quote it. Check the quote's scope as well as its strength; a SHOULD from
  a section that governs another case is not a requirement here.
- Trim narrative comments to one line of why. No issue numbers in code comments;
  they go in the commit and PR.
- Confirm the PR follows the repository conventions Detail tends to miss:
  `clippy::absolute_paths` (import, don't spell paths), `#[expect]` with a reason
  rather than `#[allow]`, methods on types rather than new single-use free
  helpers, `jiff` for time, UUID v7 ids.
- Confirm docs changed in the same PR: `CLAUDE.md`, `docs/`, and the design doc
  section 17 gaps list when the fix closes one.

### Merge mechanics

Run the full gate on the branch before calling a PR ready:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace
cargo deny check
```

Read CI status fresh with `gh pr checks <n>` at the moment you decide. Do not
reuse a run ID captured earlier in the session; runs get superseded, and a stale
failing run can hide a passing one. When a wait is unavoidable, background
`gh pr checks <n> --watch` rather than sleeping.

`main` requires signed commits and a merge queue. Detail's follow-up commits (a
rustfmt or clippy fix pushed after its CI fails) can arrive unsigned, and the
queue rejects the whole branch. Check before handing a PR over:

```bash
gh api repos/smoketurner/quack/pulls/<n>/commits \
  --jq '.[] | "\(.sha[0:8]) \(.commit.verification.verified)"'
```

Rebuild such a branch with signed commits on a new branch off `origin/main`
rather than rewriting Detail's. Amending a Detail PR means pushing to its branch;
a push to a queued PR's branch is refused, and a push while auto-merge is armed
can silently drop it from the queue. Justin merges PRs himself: report which PRs
are ready, amended, superseded, or closed, and leave the merging to him unless
he asks otherwise.

After the batch lands, re-run `make lint` and `make test` on `main`. Branches
touching disjoint files can still break each other.

## Step 7: Record the metric

Write the batch record to `.local/detail-triage-<YYYY-MM-DD>.md`: the
classification table, the review findings, and the decisions. `.local/` is
gitignored working memory; **read the most recent record before starting**.

Give the record a **Residue** section listing everything the review accepted and
did not fix, each with the finding it would become. The next pass matches new
findings against it (step 2).

A gap the review notices is residue only after the user chooses to leave it.
"Not a regression" or "already true before this PR" is an observation, not a
decision, and the next batch files it. Put such a gap on the decision list, and
follow what it admits to where that artifact is consumed before sizing it.

Then extend `references/volume-log.md`: add a row to the batch table, with the
counts from `detail-stats.py` and the judgement columns defined above the table,
and append a section with the monthly row, the classes, and what was decided.
The next run reads both to tell a trend from a blip.

Every few batches, read all the records in `.local/` together and fold any
lesson that has recurred into this skill. A lesson that lives only in a record is
not applied.

Success is **the `<30d` count and the per-class counts falling** over successive
batches. It is not an empty issue list: while Detail is still working through
backlog, a high total is expected and says nothing about the code going in.
