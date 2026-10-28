# Experiment Records

Experiment records preserve evidence that could affect future work. They are not a diary of every command.

Create a summary when a run:

- compares an approach or dataset;
- establishes or challenges a baseline;
- produces a meaningful negative or inconclusive result;
- reveals a failure mode;
- changes a technical decision; or
- would otherwise be expensive for another teammate to repeat.

A GitHub Issue is optional. If one exists, link it from the experiment summary.

## File naming

Copy `experiment-template.md` into `summaries/` using:

```text
YYYY-MM-DD-short-description.md
```

Use one summary for one coherent question. Several runs may belong in the same summary when they test the same hypothesis.

## Evidence expectations

Record enough information to interpret or reproduce the result when practical:

- objective or hypothesis;
- dataset identity, version, and split;
- code commit or explicit dirty working-tree state;
- model and configuration;
- seed, relevant environment, and hardware;
- commands or entry points;
- metrics and qualitative observations;
- artifact locations;
- validity problems and limitations;
- conclusion and recommended next step.

Do not commit large outputs or weights. Link to the team's approved artifact storage once that policy exists.

Distinguish among:

- **Positive:** evidence supports the hypothesis or meets the stated criterion.
- **Negative:** a valid run provides evidence against the hypothesis or approach.
- **Inconclusive:** the evidence is insufficient to decide.
- **Invalid:** a flaw means the result should not be used as evidence.
