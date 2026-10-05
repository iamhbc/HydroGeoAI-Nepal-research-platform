# Contributing

Thanks for helping improve HydroGeoAI-Nepal. Contributions to code, methods, documentation, accessibility, and reproducibility are welcome.

## Before you start

- For a substantial change, open an issue or discussion first so the scope and research implications can be agreed.
- Keep changes focused. Explain the problem they solve and link any related issue.
- Do not commit credentials, private station records, or other restricted data.
- The repository's default data are synthetic. Results from synthetic data can validate software and workflow behavior, but they do not establish findings about Nepal's climate.
- Before adding real observations, check the source's licence and terms, document provenance and units, and avoid publishing sensitive station locations or records.

## Research and data integrity

- Preserve the documented station and time based splits, embargoes, and leakage checks when changing model or evaluation code.
- Keep preprocessing, feature selection, and threshold selection inside the appropriate training or validation split.
- Report the data source, period, spatial coverage, missingness, and limitations alongside new results.
- Do not describe exposure screening or model output as an official forecast, warning, or estimate of risk to life.
- Do not publish datasets, models, or Hugging Face artifacts as part of a code change unless the maintainers have explicitly agreed to the release.

## Development

Use the repository's documented setup and commands in the README. For a code change, run the narrowest relevant checks first, then the broader test or reproduction command when practical. State exactly which commands you ran and any that you could not run.

Keep generated outputs, local databases, credentials, and environment files out of commits unless a change explicitly requires a small, reviewable fixture. If a result depends on a random process, use the configured seed and document any remaining nondeterminism.

## Pull requests

A pull request should include:

1. A short explanation of the problem and the change.
2. The relevant issue or research question, if one exists.
3. The checks run and their outcomes.
4. Any effect on data formats, APIs, model behavior, scientific interpretation, or reproducibility.
5. Sources and licences for any newly introduced data or external material.

Please keep claims proportional to the evidence and call out known limitations. Maintainers may request changes before merging.
