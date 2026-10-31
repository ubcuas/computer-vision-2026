# Contributing

This repository uses a lightweight workflow intended to preserve useful knowledge without requiring an Issue for every investigation.

## Starting work

Work may begin from a GitHub Issue, a direct teammate request, a meeting action item, or an exploratory investigation.

Before making substantial changes:

1. Read `AGENTS.md` and the relevant documentation under `docs/`.
2. Define the immediate question or outcome.
3. Identify important assumptions and unknowns.
4. Decide what evidence would make the work useful to the next teammate.

Create or link a GitHub Issue when the work is shared, long-running, blocked, decision-heavy, or needs coordination. A quick investigation does not need to wait for an Issue.

## Branches and pull requests

- Use a short, descriptive branch name, such as `research/deer-datasets` or `experiment/tag-ocr`.
- Keep changes focused enough to review and verify.
- Open a pull request when the work is ready for team review.
- In the pull request, explain the objective, evidence, limitations, unresolved questions, and next step.
- Update durable documentation when a change affects competition interpretation, datasets, experiments, or an accepted technical decision.

## Experiments

Follow `docs/experiments/README.md`. Preserve negative and inconclusive results when they can prevent repeated work or change a future decision.

Do not commit large datasets, recordings, model weights, credentials, secrets, or generated outputs unless the repository explicitly adopts a suitable policy for them.

## Review expectations

- Claims should be supported by the official competition sources, code, or recorded experimental evidence.
- Competition interpretations and material architecture, dataset, or evaluation changes require human review.
- Report checks that were actually run; do not imply that unrun checks passed.
- Keep unrelated changes out of the pull request.
