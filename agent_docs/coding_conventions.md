# Coding Conventions

## Style
- Black formatter, line length 88
- isort for import ordering
- Type hints everywhere: `def aggregate(gradients: list[torch.Tensor], f: int) -> torch.Tensor:`
- NumPy-style docstrings on public functions

## Research code standards
- Every function that involves randomness must accept a `seed` param
- No magic numbers — define constants at module level with descriptive names
- Tensor operations: always specify device explicitly
- Memory: call torch.cuda.empty_cache() between experiments on GPU
- Timing: use time.perf_counter(), not time.time()

## File organization
- One class per file for attacks and defenses
- Shared utilities go in src/utils/
- Experiment configs are data, not code — always JSON
- Notebooks are for analysis only, never for running experiments
