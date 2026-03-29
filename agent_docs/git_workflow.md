# Git Workflow

## Branches
- main: protected, requires passing tests
- dev: integration branch, merge features here first
- feature/{name}: new functionality
  - feature/backdoor-attack
  - feature/reputation-defense
  - feature/gpt2-wrapper
  - feature/akash-deployment
- experiment/{name}: experiment runs (may contain large result files)
  - experiment/cifar10-baseline
  - experiment/scalability-timing
  - experiment/akash-gpt2xl
- docs/{name}: documentation and paper changes

## Commit message format
- feat: new feature (feat: add backdoor attack implementation)
- fix: bug fix (fix: correct gradient dimension in Krum timing)
- exp: experiment results (exp: CIFAR-10 Krum vs backdoor alpha=0.5)
- docs: documentation (docs: add experiment protocol)
- test: tests (test: add unit tests for ASR metric)
- refactor: code improvement (refactor: extract metrics to utils)

## Rules
- Commit early, commit often — small commits are better than large ones
- Always include the experiment config hash in experiment commits
- Never force push to main or dev
- Results files (.json in results/) are committed but not modified
- Large model checkpoints go in .gitignore
