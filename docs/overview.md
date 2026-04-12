# Overview

## Start Here

For new users, ignore the legacy path at first.

1. Install the repo in editable mode.
2. Run `python examples/quickstart_v2.py`.
3. Run `python ta_experiments.py` if you want the fuller diagnostics comparison.

## Recommended Path

- Optimizer: `TopologicalAdamV2`
- Diagnostics helper: `topological_adam.analysis`
- Stopping helper: `topological_adam.stopping.ReconnectionStoppingRule`

## Experimental Path

- Optimizer: `TopologicalAdamSDS`
- Purpose: test the SDS-inspired two-temperature efficiency gate without changing the default repo path
- Current status: stable, but not yet clearly better than V2

## Legacy Path

- Optimizer: `TopologicalAdam`
- Purpose: provenance and comparison only

## What To Expect From The Main Comparison

The supported experiment compares:
- a control run with the auxiliary correction disabled
- a V2 run with the auxiliary correction active

This tells you whether the internal signals are:
- finite
- stable
- interpretable
- plausibly useful as monitoring signals

It does **not** prove that the optimizer is superior in a benchmark sense.
