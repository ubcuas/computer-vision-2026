# UBC UAS Computer Vision 2026-2027

Computer-vision research and development for the UBC Uncrewed Aircraft Systems team's 2027 Aerial Evolution Association of Canada student competition entry.

The current focus is Task 1, the Herd Survey: detect and count deer decoys, group them into spatial clusters, read ear-tag IDs, identify anomalous interactions with human artifacts, and support timely generation of the competition report.

## Start here

- [`AGENTS.md`](AGENTS.md): durable repository instructions for teammates and AI tools.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): lightweight contribution and handoff workflow.
- [`docs/README.md`](docs/README.md): documentation map explaining what each file owns and when to update it.
- [`docs/competition/task-1.md`](docs/competition/task-1.md): confirmed Task 1 requirements and the current CV decomposition.
- [`docs/competition/ambiguities.md`](docs/competition/ambiguities.md): unresolved competition and engineering questions.
- [`docs/experiments/README.md`](docs/experiments/README.md): how to preserve useful experimental evidence.
- [`docs/datasets/README.md`](docs/datasets/README.md): dataset documentation expectations.

## Repository layout

- `dataloaders/`: reusable loaders for core datasets so training and evaluation use consistent behavior.
- `prototyping/`: individual model prototypes and early investigations.
- `docs/competition/`: competition sources, interpretations, and open questions.
- `docs/experiments/`: durable summaries of meaningful successful, negative, and inconclusive experiments.
- `docs/datasets/`: dataset provenance, licensing, versions, splits, and limitations.
- `docs/decisions/`: consequential technical decisions and their rationale.

Large datasets, flight recordings, trained weights, and generated outputs should not be committed directly unless the team adopts an appropriate storage policy.

## GFL prototype

See [GFL baseline notebook and server handoff](prototyping/gfl-deer/BASELINE.md). Training is manually launched; datasets and model outputs remain external.
