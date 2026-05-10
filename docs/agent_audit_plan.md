# Agent Audit Report Plan

## Purpose

The next improvement should make a completed fit-control loop easier to inspect. Today the loop leaves JSON records, plots, and gate metrics, but a human still has to reconstruct the story from several files. This plan adds a small audit layer that produces a readable report without changing the physical fitter, refit gate, or existing tool semantics.

## Scope

Phase 1 delivers:

- `audit_report.md`: a human-readable summary for one loop run.
- `audit_report.json`: the same information in a stable machine-readable shape.
- `fit_control_loop.audit_report_paths`: paths attached to the existing loop summary.
- A bounded experiment budget: `max_rounds` is the default number of complete experiment rounds, while `hard_max_rounds` is the absolute cap.
- A two-step round: the agent first decides an experiment, the harness executes/refits, then the agent assesses the result in the same round and decides whether another round is justified.
- `request_more_budget`: an assessment-step tool that can ask the harness to continue after the default budget if it has a concrete next experiment.
- Tests proving reports are emitted for no-refit and refit loop exits.

Phase 1 explicitly does not rewrite the loop into a full session abstraction, change gate thresholds, or introduce template dependencies.

## Design

The audit layer consumes data already produced by the loop:

- final review record
- `fit_control_loop` history
- final fit summary
- candidate output paths
- stop reason

It derives experiment-oriented fields:

- `overall_status`: mapped from stop reason and final fit quality.
- `executive_summary`: short factual summary of the loop result.
- `key_metrics`: fit RMS, reduced chi-square, component count, quality.
- `rounds`: each round's control status, tool count, decision, warnings, and reasons.
- `findings`: structured highlights from gate decisions and warnings.
- `human_action_items`: concrete checks for the reviewer.
- `budget_events`: approved or denied agent budget requests.

Reports are written under:

```text
<output_dir>/<sample_id>.audit/
├── audit_report.md
└── audit_report.json
```

This keeps audit output next to generated review packets while avoiding changes to existing packet file names.

## Implementation Steps

1. Add `src/astroagent/agent/audit/` with a dependency-free renderer.
2. Add `request_more_budget` to the fit-control tool schema and patch normalization allow-list.
3. Teach `run_fit_control_loop` to treat `max_rounds` as a soft default budget and `hard_max_rounds` as the real cap.
4. Add an in-round assessment call after each refit. This call receives the just-produced refit metrics and images, writes a short assessment rationale, and may call `request_more_budget`.
5. Feed assessment and budget decisions back through `fit_control_last_refit_feedback` so the next experiment round can use them.
6. Call the renderer at every `run_fit_control_loop` return path.
7. Attach report paths to `final_record["fit_control_loop"]["audit_report_paths"]`.
8. Print the report path from the fit-loop CLI.
9. Add tests for budget extension, report JSON/Markdown existence, and basic content.

## Later Phases

Phase 2 can add explicit annotate/terminal tools such as `record_finding`, `flag_uncertainty`, and `emit_audit_report`.

Phase 3 can introduce inspect tools for residuals, continuum anchors, and posterior bands.

Phase 4 can replace implicit record-carried state with an `AgentSession` that can checkpoint and export SFT-style message traces.
