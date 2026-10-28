# Documentation Guide

This is the map for the repository's shared knowledge base. It explains what each document owns, when people and AI tools should read it, and when it should be updated.

The goal is to preserve useful knowledge without turning every investigation into paperwork. A GitHub Issue can coordinate work, but it is not required before research begins or before useful findings are documented.

## Quick start for a new teammate

1. Read the root `README.md` for the project purpose and repository layout.
2. Read `CONTRIBUTING.md` for the lightweight working and review process.
3. Read `competition/task-1.md` for the current Task 1 requirements and CV workstreams.
4. Check `competition/ambiguities.md` before relying on an interpretation of an unclear rule.
5. Use the file map below to find the relevant dataset, experiment, or decision context.

## Quick start for an AI tool

1. Read the applicable `AGENTS.md` instructions first.
2. Read the files relevant to the immediate task; do not load or rewrite the entire knowledge base without a reason.
3. Distinguish official facts, team decisions, interpretations, hypotheses, and unknowns.
4. Before handoff, apply the materiality test below and update only the canonical files affected by the work.
5. Report which knowledge files changed and which unresolved questions remain.

## File and directory map

### Repository-level guidance

| Location | What it owns | Read it when | Update it when |
| --- | --- | --- | --- |
| `/README.md` | Project purpose, top-level structure, and entry points | First entering the repository | The repository's purpose, major directories, or primary navigation changes |
| `/AGENTS.md` | Durable repository-wide instructions and boundaries for AI-assisted work | Before an AI tool performs repository work | A rule should apply broadly to future AI work; keep it concise and avoid project facts owned elsewhere |
| `/CONTRIBUTING.md` | Human contribution, branch, review, experiment, and handoff workflow | Beginning shared work or preparing a pull request | The team's normal collaboration process changes |
| `/.gitignore` | Local files, secrets, datasets, generated outputs, and model artifacts that must not enter ordinary Git history | Before adding a new artifact type or data directory | A tool creates a new class of local output, or the team intentionally changes its artifact-storage policy |
| `/docs/README.md` | Documentation ownership, routing, and maintenance rules | Locating or updating shared knowledge | A knowledge file is added, removed, renamed, or given a different responsibility |

### Competition knowledge

| Location | What it owns | Read it when | Update it when |
| --- | --- | --- | --- |
| `/docs/competition/README.md` | How official sources and team interpretations are separated | Adding or interpreting competition information | The competition-document workflow or source hierarchy changes |
| `/docs/competition/sources/` | Unmodified copies of official competition documents | Exact official wording or figures matter | A new official version or supplemental document is published; add a new file and never rewrite an existing source |
| `/docs/competition/source-register.md` | Source provenance, version, integrity, and current/superseded status | Verifying whether competition information is current | An official source or written organizer clarification is added, checked, revised, or superseded |
| `/docs/competition/task-1.md` | Concise confirmed requirements, scoring context, CV decomposition, and research direction | Working on Task 1 design, data, models, evaluation, or reporting | Confirmed requirements, reviewed implications, workstreams, or well-supported research direction changes |
| `/docs/competition/ambiguities.md` | Unresolved rules questions and engineering unknowns | A rule or constraint is unclear | A question is discovered, clarified, resolved, or materially reframed; preserve the evidence and resolution type |

### Datasets, experiments, and decisions

| Location | What it owns | Read it when | Update it when |
| --- | --- | --- | --- |
| `/docs/datasets/README.md` | Dataset candidates, provenance, licenses, formats, splits, limitations, and approval status | Considering, downloading, transforming, or evaluating a dataset | Material dataset information is learned or a dataset becomes a shared dependency |
| `/docs/experiments/README.md` | Rules for deciding which experiments deserve durable records | Planning or completing a meaningful experiment | The experiment-recording convention changes |
| `/docs/experiments/experiment-template.md` | Standard structure for reproducible experiment summaries | Creating a new experiment summary | The team agrees that every future experiment record needs a new field |
| `/docs/experiments/summaries/` | Evidence and conclusions from meaningful positive, negative, inconclusive, or invalid experiments | Comparing approaches or avoiding repeated work | A run produces evidence that can affect later work or decisions |
| `/docs/decisions/README.md` | Criteria and template for technical decision records | Making a consequential or difficult-to-reverse decision | The decision-record convention changes |
| `/docs/decisions/NNNN-short-title.md` | One accepted, proposed, or superseded technical decision and its rationale | Work depends on that decision | The decision status changes; supersede rather than silently rewriting accepted history |

### Collaboration templates

| Location | What it owns | Read it when | Update it when |
| --- | --- | --- | --- |
| `/.github/pull_request_template.md` | Minimum handoff information for review | Preparing or reviewing a pull request | Reviewers repeatedly need information the template does not request, or a field creates unnecessary friction |
| `/.github/ISSUE_TEMPLATE/rules-question.yml` | Optional structured intake for consequential competition-rule questions | A rules ambiguity needs shared tracking or organizer follow-up | The information needed to resolve rules questions changes |

### Code and configuration areas

| Location | What it owns | Update guidance |
| --- | --- | --- |
| `/dataloaders/` | Reusable loading and conversion code for shared datasets | Prefer shared, tested behavior over copying loader logic into individual prototypes |
| `/prototyping/` | Bounded model prototypes and exploratory implementation | Keep each investigation understandable; promote reusable code only after its interface and evidence are sufficiently stable |
| `/configs/experiments/` | Version-controlled experiment configurations once the team begins using them | Create configurations that can be tied to code, data, and an experiment summary; do not create empty configuration files |

The `configs/experiments/` directory is a planned location and does not need to exist until the first real configuration is added.

## Materiality test

Update the knowledge base when omitting a result could reasonably:

- cause another teammate to repeat expensive discovery or a failed approach;
- change an implementation, architecture, dataset, or evaluation decision;
- cause an incorrect interpretation of the competition rules;
- hide a limitation, blocker, risk, or unresolved assumption;
- make an experiment difficult to interpret or reproduce; or
- affect a future teammate's ability to continue the work.

Do not create durable documentation for routine command output, step-by-step narration, harmless temporary errors, abandoned thoughts with no downstream value, or facts already maintained in another canonical file.

## Updating documentation well

1. Update the file that owns the information instead of copying it into several places.
2. Preserve provenance: link the official source, code, configuration, experiment, decision, Issue, or pull request when available.
3. Label uncertain information as an interpretation, hypothesis, or unknown.
4. Record useful negative and inconclusive evidence, not only successes.
5. Preserve history when it matters. Add new official source versions and superseding decisions rather than overwriting the old evidence.
6. Keep summaries concise and link to detailed evidence.
7. Update documentation in the same pull request as the work that made it inaccurate, when practical.
8. Ask for human review when changing a competition interpretation, dataset policy, evaluation method, or consequential technical decision.

## Adding a new knowledge file

Before adding another document, confirm that its purpose is not already owned by an existing file. Give the new file one clear responsibility, link it from the appropriate index, and add it to this map. Avoid creating empty folders or speculative templates that the team is not yet using.
