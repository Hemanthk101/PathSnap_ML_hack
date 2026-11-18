# ML & Optimization

## Your responsibility
Model comparison, Bayesian tuning, grouped validation, metrics and model artifacts.

Primary ownership: pathsnap/experiments.py, models/, results/

## Shared project
This archive contains the same project source and model artifacts as a shared starting point. Coordinate changes to shared files; integrate all four contributions into one project. See README.md for setup and data import. Data is not bundled: obtain the approved dataset and import it using scripts/import_dataset.py.

## Integration
Person 1 supplies candidate features and schema. Person 2 consumes those features and supplies model artifacts and inference scores. Person 3 consumes scores and the directed road graph and returns the decoded route. Person 4 connects inference to the API and UI. Preserve existing interfaces or agree changes with the team. Keep evaluator references private and outside training and tuning.

Record changed files, dependencies, commands, results and limitations for your handoff.
