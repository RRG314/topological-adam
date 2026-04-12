# Roadmap

## Near Term

1. Add a small benchmark harness beyond the current synthetic diagnostics setup.
2. Test whether the `J_t` stopping rule transfers to real datasets and small vision models.
3. Measure when the topological correction helps relative to a matched control and when it does not.
4. Decide whether the SDS-inspired branch deserves further tuning or should remain a clearly archived experimental fork.

## Mid Term

1. Tighten the interpretation of `J_t` so it is separated cleanly from generic gradient-magnitude effects.
2. Add exportable reports for diagnostics runs.
3. Build a better bridge document showing what can and cannot be carried over from the MHD theory repo.
4. Test whether bounded thermal-gating ideas help on harder workloads or simply reparameterize V2.

## Longer Term

1. Determine whether there is a reproducible workload class where V2 offers a real optimization advantage.
2. Prove or disprove stronger convergence statements instead of relying on qualitative narratives.
