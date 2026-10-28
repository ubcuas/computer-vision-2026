# Repository Guidance

## Project context

- This repository contains computer-vision work for the UBC Uncrewed Aircraft Systems team.
- The current competition focus is Task 1 of the 2027 Aerial Evolution Association of Canada student competition: surveying deer from a UAV.
- The current known rules baseline is the 2027 AEAC CONOPS v1.0 dated 2026-09-08. Verify competition-critical details against the latest official document and organizer clarifications before relying on them.
- Put individual model prototypes under `prototyping/`.
- Put reusable dataset loaders under `dataloaders/` so experiments use consistent loading and evaluation behavior.

## Repository context

- Read `README.md` for repository orientation.
- Read `docs/README.md` for the documentation map, ownership boundaries, and update rules.
- For Task 1 work, read `docs/competition/task-1.md` and check `docs/competition/ambiguities.md` for unresolved questions.
- Use `docs/competition/source-register.md` to verify the version and provenance of competition information.
- Follow `docs/experiments/README.md` when a run produces evidence worth preserving.
- Follow `docs/datasets/README.md` before adding or using a dataset.
- Record consequential, durable technical decisions using `docs/decisions/README.md`.

## Sources of truth

Use the following order when sources disagree:

1. The latest official competition documents and written organizer clarifications.
2. Versioned repository documentation and accepted team decisions.
3. Reproducible experiment evidence tied to code, data, and configuration versions.
4. Working notes, interpretations, and AI-generated suggestions.

Label important statements as confirmed facts, team decisions, interpretations, hypotheses, or unknowns. Do not turn an assumption into a competition requirement.

## Starting work

Work may begin from any of these sources:

- a GitHub Issue;
- a direct teammate request;
- an action item from a meeting; or
- a small exploratory investigation or experiment.

Before making changes, understand the immediate objective, inspect relevant code and prior work, state material assumptions or unknowns, and choose a lightweight success check appropriate to the task.

A GitHub Issue is helpful when work is shared, long-running, blocked, connected to a decision, or likely to produce evidence the team will need later. Do not block a quick investigation merely because no Issue exists. If exploratory work becomes substantial, create or link an Issue during handoff when authorized.

## Experiments and evidence

For a meaningful training or evaluation run, preserve enough information for another teammate to understand or reproduce it when practical:

- purpose or hypothesis;
- dataset identity and train/validation/test split;
- code commit or working-tree state;
- model and configuration;
- random seed, relevant environment details, and hardware when they affect results;
- metrics and important qualitative observations;
- output or artifact location; and
- conclusion, including negative or inconclusive results.

Distinguish a valid negative result from a failed or invalid run. Avoid data leakage, and report the metrics and evaluation slices relevant to the stated objective.

Do not commit large datasets, flight recordings, trained weights, generated outputs, credentials, or secrets unless the repository explicitly adopts a suitable storage policy for them.

## Knowledge maintenance

Before handoff, update the appropriate canonical file when the work produced information whose omission could cause repeated work, a different technical decision, an incorrect competition assumption, or an irreproducible result. Follow the detailed routing and update rules in `docs/README.md`.

- New or revised official competition source: update `docs/competition/source-register.md`; never edit a preserved source file.
- Competition requirement or reviewed Task 1 implication: update `docs/competition/task-1.md`.
- Unresolved or resolved competition question: update `docs/competition/ambiguities.md` and link its evidence.
- Dataset provenance, license, quality, split, or suitability finding: update `docs/datasets/README.md`.
- Meaningful experiment, including a useful negative or inconclusive result: add a summary under `docs/experiments/summaries/`.
- Consequential technical decision: add or supersede a record under `docs/decisions/`.
- Repository workflow or contributor expectation: update `README.md`, `CONTRIBUTING.md`, or `AGENTS.md`, according to the ownership described in `docs/README.md`.

Do not record routine commands, transient errors, narration, or speculation with no future value. Link to canonical evidence instead of copying it. A GitHub Issue is not required before updating the knowledge base.

## Verification and handoff

- Run the narrowest relevant checks and report what was actually observed. Never claim an unrun check passed.
- Summarize what changed or was learned, the supporting evidence, limitations, unresolved questions, and the most useful next step.
- Link an Issue, pull request, or experiment record when one exists; their absence is not itself a blocker.
- Record failed approaches when they could prevent duplicated work or inform the next attempt.
- Promote information into durable documentation only when it is sufficiently stable or clearly marked as provisional.
- Leave the repository understandable and reproducible enough for the next teammate or AI tool to continue.

## Safety and collaboration boundaries

- Preserve teammates' work and avoid unrelated changes.
- Ask for human review when competition rules are ambiguous or a decision materially changes the system architecture, dataset policy, or evaluation method.
- Do not delete data, modify external systems, publish results, merge, commit, or push unless the user explicitly requests it.
- Treat AI output as a proposal until it is supported by repository evidence, experimental results, or human review.
