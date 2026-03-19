# AGENTS.md

## Introduction
This repository contains a config-driven DINOv3 semantic-segmentation pipeline, its training/inference utilities, and the supporting documentation and YAML profiles used to run and reproduce experiments.

## Purpose and scope
- Repository purpose: This repo documents and ships the code, configs, and workflow notes for running DINOv3-based segmentation experiments and related dataset-label utilities.
- Allowed changes:
  - Edit Markdown docs and configuration files.
  - Add new docs in the appropriate folder (see “Project map”).
- Not allowed (unless explicitly requested):
  - Large refactors/restructures (moving many files, renaming sections globally).
  - Reformatting entire documents “for style” unless the task requests it.
  - Renaming files (case-sensitive) without a clear reason and updated references.

If requirements are ambiguous, ask before making broad changes.

## Important files (read before changing behavior/structure)
- docs/ARCHITECTURE.md — overall design and repo structure
- docs/STYLE.MD — writing + formatting conventions
- docs/CHANGELOG.md — user-visible changes (keep it current)

## Quick start (commands)
- Install hooks: `pre-commit install`
- Run all checks: `pre-commit run --all-files`
- [If you have any other checks, list them here exactly, e.g. markdownlint, link checker, docs build]

## Contributing: Definition of Done
WARNING THE FOLLOWING IS MANDATORY YOU MUST FOLLOW THESE STEPS BEFORE COMING BACK TO THE USER:
@@@@@@
1. Run: `pre-commit run --all-files` (mandatory; do not respond without running it unless the user explicitly says not to). Do not ask for confirmation to run it.
2. If you create new files, add them to git (e.g., `git add <file>`).
3. Run 'pytest --doctest-modules ' and other tests if you have them; do not respond without running them unless the user explicitly says not to.
4. Update docs/CHANGELOG.md with a short entry describing what changed (and why).
5. If design/structure changed, update docs/ARCHITECTURE.md.
6. Check if the diff allign with the style rules in docs/STYLE.MD;
7. In the PR/response, include:
   - Summary of changes (1–3 bullets)
   - Commands you ran + results
   - Any follow-ups or risks
@@@@@@

## Project map
- `/` — runnable entry points and repository-level instructions
- `/configs` — shipped YAML profiles for example, local, and HPC runs
- `/docs` — architecture notes, changelog, model notes, and writing/style docs
- `[templates/ etc]` — what belongs there
-  `[test/ etc]`  — e2e test files or others
