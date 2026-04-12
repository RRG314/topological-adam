# SDS Candidate Branch

## What It Is

`TopologicalAdamSDS` is an experimental third branch that keeps the V2 auxiliary-field dynamics but gates the topological correction with a bounded two-temperature efficiency.

The design is inspired by the SdS research program's two-temperature / Carnot-efficiency language:
- compute one positive scale from each auxiliary field
- order them as `T_hot >= T_cold`
- define a bounded efficiency `eta_C = 1 - T_cold / T_hot`
- use that efficiency to modulate the correction strength

This keeps the branch self-contained. It does **not** depend on the SDS repository at runtime.

## Why It Was Added

You asked for a serious pass on whether there is a clean SDS-informed version that is actually worth testing.

This is the cleanest version I could justify without faking a deeper connection than the code supports.

## Current Status

- stability: supported on the current small-task benchmark suite
- novelty claim: not made
- default status: **not** the default path
- recommendation: keep as an experimental branch, not as the main optimizer

## Benchmark Result

The current benchmark report is in:
- `docs/sds_candidate_benchmark.json`

Summary of the current five-seed run:
- the benchmark now includes `Adam` as a baseline
- quadratic: `Adam` is best; SDS improves on V2 but does not beat the baseline
- linear regression: V2 is slightly best, SDS is close behind, and Adam is slightly worse
- XOR: V2 and SDS both reach perfect mean accuracy and are effectively tied
- clustered classification: all three reach perfect mean accuracy, with only negligible final-loss differences

The important conclusion is not that SDS is better. It is that the branch is:
- real
- runnable
- stable
- competitive enough to preserve, but currently neutral rather than clearly superior

So this branch survives as an experiment, but it is **not yet justified as the new main version**.

## Honest Interpretation

This is exactly the kind of branch that belongs in the repository if it is labeled correctly:
- good enough to preserve
- not good enough to replace V2
- worth revisiting only if future tasks show a real advantage
