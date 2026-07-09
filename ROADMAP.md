# Roadmap

This roadmap is intentionally narrow. The package is being prepared as a
maintainable optimizer library, not as a collection of speculative research
notes.

## Near term

1. Keep `TopologicalAdamV3` as the recommended optimizer and preserve exact
   Adam/AdamW reduction tests.
2. Keep `TopologicalAdamV4` experimental, with claims limited to
   loop-structured or oscillatory trajectories.
3. Add independent benchmarks on additional real workloads before broadening
   any performance claims.
4. Improve CPU/GPU wall-clock measurements for V3 and V4.

## Maintenance

1. Keep benchmark scripts reproducible and store the result JSON used in the
   paper.
2. Keep public docs focused on installation, API behavior, tests, and bounded
   evidence.
3. Preserve legacy V1/V2/SDS behavior through tests, but avoid promoting those
   paths as the main user story.

## Longer term

1. Explore `foreach` or fused update paths if benchmarks show overhead is a
   practical blocker.
2. Develop convergence analysis for the gated V3 update.
3. Revisit V4 only if additional loop-structured optimization tasks show
   consistent benefit over tuned Adam and V3.
