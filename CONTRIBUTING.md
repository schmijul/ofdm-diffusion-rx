# Contributing

Thanks for contributing to this repository.

## Quick Start

1. `make venv`
2. `make install`
3. `make doctor`
4. `make quick-test`

## Common Workflows

- Main fast study: `make regime-study-fast`
- Prior trend study: `make prior-sweep`
- Long overnight study: `make regime-study-large`

For long runs, you can defer plotting to reduce runtime:

- `make regime-study-large SKIP_PLOTS=1`
- `make prior-sweep SKIP_PLOTS=1`

## Development Expectations

- Keep experiments reproducible through config files.
- Prefer adding tests for behavior changes.
- Run `make quick-test` before committing.
- Use small, coherent commits with descriptive messages.

## Result Conventions

- Delta columns use `Diffusion - MMSE`.
- Negative delta means diffusion performed better than MMSE.
- Benchmark CSV keeps both names for compatibility:
  - `delta_diff_minus_mmse_*`
  - `diffusion_gain_vs_mmse_*`
