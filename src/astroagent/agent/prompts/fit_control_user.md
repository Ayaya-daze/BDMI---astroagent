Task: fit_control. Inspect the wavelength-space overview image first, then the transition-frame fit image.

## Line-Family Context

Use input.line_family_context as soft physical background. If the line is a doublet or multiplet, sibling agreement at similar velocity is supporting evidence, but mismatch is not an automatic failure.

Preserve and report single-member or inconsistent features as possible blends, contamination, bad pixels, low S/N, saturation, edge effects, or ambiguous real cases; do not delete or mask them solely because the sibling panel is weaker or absent.

## Continuum Triage

First decide whether the continuum is trustworthy across the target doublet window. Check whether the fitted continuum follows uncontaminated flux on both sides of the absorption complex, whether saturated troughs or doublet/blend edges have pulled the continuum down, and whether continuum anchors/masks are missing or biased.

If the continuum is visibly wrong, prioritize continuum tools such as update_continuum_mask, add_continuum_anchor, or remove_continuum_anchor before changing absorption sources. In that case, do not add or retune sources merely to absorb a continuum error; request a refit after the continuum edit so the next round can judge sources on the corrected normalization.

For continuum edits, use plot_data.continuum_anchor_table: anchor_index identifies the displayed current anchor, and wavelength_A is the stable anchor identity across refits. When calling remove_continuum_anchor, include both anchor_index and wavelength_A from the table whenever possible.

If the continuum is wrong because existing anchor points sit on saturated absorption, remove those biased anchor indices first. Do not add a new anchor at a wavelength that already has a high clean anchor nearby; that usually has little effect. Only add_continuum_anchor on visibly clean sideband flux where no adequate anchor exists, and set continuum_flux to the local unabsorbed flux level, not to a trough.

Use update_continuum_mask narrowly to exclude absorption-dominated pixels from future anchor selection; do not use a broad continuum mask as a substitute for removing specific bad low anchors. Prefer one or two high-confidence anchor removals over a wide mask when the bad continuum is caused by low anchor_flux values inside the trough.

## Decision Policy

Decide which fitting tools to call. If fit_summary.quality is good and agent_review.required is false, prefer no_action unless the images show a clear residual, missing component, bad continuum anchor, or masked-pixel problem that the structured metrics missed.

If the fit is inspect, saturated, broad, asymmetric, or visibly under-modeled, propose at least one concrete exploratory edit such as add_absorption_source, update_absorption_source, set_fit_window, set_fit_mask_interval, or continuum anchor/mask edits.

If no edit is safe, return JSON with task='fit_control', status='no_action', tool_calls=[], and a rationale.

## Feedback From Previous Rounds

If fit_control_context.last_rejected_refit is present, treat it as feedback that the refit may have worsened, not as proof the previous idea was useless. Inspect those non-mainline refit images after the current-state images. Decide whether the previous edit exposed a useful partial fix, such as a corrected continuum followed by a bad source refit. In that case, keep the useful edit concept and add a follow-up mask/remove/window edit for the failure mode.

If fit_control_context.last_refit_feedback is present and its evaluation warnings include continuum_changed_fit_not_directly_comparable or continuum_refit_needs_source_mask_followup, the continuum change is already part of the current state. Do not repeat the same continuum-only anchor removals; the next useful action must be a source, fit-mask, fit-window follow-up, or explicit human-review/no-action rationale.

If the feedback includes edits_in_overlapping_velocity_frames, the previous round edited pixels that are shown in more than one transition velocity panel. Use fit_control_evaluation.metrics.delta.overlap_edit_policy.edit_annotations to inspect all affected panels before the next edit. This is a general wavelength-overlap rule, not only a doublet rule.

If the feedback includes overlap_affected_velocity_frames_unaddressed, the previous round edited one velocity frame but left another frame that sees the same observed-wavelength pixels without any direct tool action. Inspect and handle the listed unaddressed_affected_transition_line_ids before continuing source additions.

If the feedback includes overlap_affected_velocity_frames_without_fit_pixels, the affected panel may have no residual samples, so do not infer that it is fine from RMS; use the plot and add a mask/window/source edit or leave it for human review.

If the feedback includes missing_overlap_mask_after_source_removal or missing_overlap_mask_after_source_update, the previous round changed a source inside an observed-wavelength overlap without masking the shared pixels. Fix that specific overlap with a tight set_fit_mask_interval before adding more sources.

If the feedback includes fit_mask_boundary_left_residual_in_fit_pixels or high_context_diagnostic_residual, a previous mask/window left contaminating absorption near the source work window or in diagnostic context. Inspect the affected velocity range and widen/shift the mask, tighten the fit window, or leave the case for human review; do not trust a broad source that was pulled by those residuals.

If the feedback includes global_fit_rms_not_directly_comparable, compare per-transition residual panels instead of treating the global RMS as a same-pixel score.

If the feedback includes continuum_changed_fit_not_directly_comparable, the continuum edit may be scientifically right even though the old source seeds now fit worse. Do not simply repeat the same continuum-only anchor removals; use the non-mainline refit image as the current normalization hypothesis and add the needed source, mask, or window follow-up, or leave it for human review.

For a refit that stayed out of the main state, explicitly attribute the worse fit to the previous round's tool calls before proposing anything new. If that patch added sources or masks and RMS/chi2 worsened, treat those source/mask calls as high-risk negative feedback. Do not repeat the same source/mask interpretation, and do not make a nearby variant of it, unless there is new image evidence in the current accepted state that was absent before.

## Source And Window Policy

Saturated or strong systems are not automatically better with more components: if the core is saturated, broad, or physically degenerate, prefer one narrow, testable edit or no_action with human-review rationale over many sources that only trace noise.

The blue Voigt component curves are normalized transmission curves, not additive absorption depths; the red combined model multiplies transmissions. A new source placed on pixels that are already saturated by another component may not make the combined curve visibly explode, but it is still a degenerate/redundant interpretation and should be avoided unless the overview shows an independent target feature.

The source-picking work window is fit_control_context.source_work_window_kms, usually [-400, +400] km/s. The inner core is fit_control_context.source_core_window_kms, usually [-360, +360] km/s; residuals near the work-window boundary are a boundary band, not pure context.

Only add/update/remove absorption sources inside the work window. By default the fitter also fits Voigt source parameters inside this work window; pixels outside it are context diagnostics for contamination and continuum decisions, not primary source-fitting targets. Use set_fit_window only when you intentionally want to expand the source-fit window. Spend most source-edit budget on the core before boundary/context cleanup.

Before requesting refit, ask whether the proposed edits are likely to improve the scientific interpretation of the target complex.

## Residual And Overlap Policy

In saturated doublets, apparent far/context absorption can be the projected sibling transition or blend structure in the current transition frame. A narrow mask or window edit for that projected sibling/blend is valid when the image evidence is clear, but explain it explicitly as sibling/blend control rather than generic cleanup.

Do not over-trust residuals alone. A large negative residual means the current model misses flux, but it does not mean the missing feature is a target absorption source for that velocity frame. Before add_absorption_source, verify in the wavelength overview that the feature belongs to the target transition rather than another line, a projected overlap, a boundary contaminant, or a continuum artifact.

For overlap/context/boundary residuals, prefer a narrow mask or human review over adding a source unless the overview clearly supports target absorption.

For residual_hints outside fit_control_context.source_core_window_kms, especially when in_velocity_frame_overlap=true or preferred_action='set_fit_mask_interval_or_human_review', the prior should favor set_fit_mask_interval or human review over add_absorption_source. Adding a source outside the core is a higher-risk exception that requires clear wavelength-space evidence that this is the target transition, not just a red residual.

Different transition-frame velocity plots are coordinate transforms of the same wavelength data and may overlap in wavelength space. Use fit_control_context.diagnostic_hints.velocity_frame_wavelength_overlaps to compare each velocity panel's wavelength interval and overlap region before deciding whether an edge feature is a separate source, a projected feature from another transition frame, or a blend.

When you identify a far-edge or boundary feature as projected/blend contamination in one transition frame, audit every other transition frame that shares the same observed-wavelength pixels before refit. Use fit_control_context.diagnostic_hints.sibling_work_window_projections when present as a convenient mapping, but apply the same rule to any listed velocity-frame wavelength overlap.

If the corresponding visible edge also biases the fit, prefer a tight set_fit_mask_interval over removing an otherwise useful source; if you leave it unmasked, state why in the rationale.

If you remove or move a fitted source because it was actually projected/blend absorption, also add a tight set_fit_mask_interval over that same observed-wavelength overlap in the affected transition frame unless you can see that the pixels no longer enter the fit. Deleting the source without masking the overlap usually exposes a large negative residual at the velocity-frame crossing instead of fixing the model.

Velocity panels are not independent evidence: a trough may be the same observed-wavelength pixels shown in multiple frames. If you mask, remove, or update a projected/blend edge in one transition frame, also decide what happens in every other affected frame in the same round: add a corresponding tight mask/window/source edit when the plot supports it, or explicitly state that the affected frame should be left for human review. Do not assume the other panel is fine just because it has no residual plot or no fitted sources.

## Mask Policy

A mask only changes which pixels constrain the next fit; it does not move a component that was previously pulled toward the masked edge. After any set_fit_mask_interval, inspect fitted sources in the same transition frame. If a component is centered in or near the masked/edge/blend feature, is very broad, or is pinned to a parameter bound, pair the mask with update_absorption_source or add_absorption_source so the remaining target component is seeded on the real core/shoulder. If you intentionally leave source positions unchanged, explain why the existing component is already centered on unmasked target absorption.

A fit mask must cover the whole contaminating/blend feature with a modest boundary margin. A too-narrow mask that leaves the neighboring absorption edge in unmasked fit pixels is worse than no mask: those residual pixels still participate in the fit and can pull the Voigt source broad or off-center. Before requesting refit, check both mask edges on every affected panel and widen or shift the mask if the adjacent trough/wing is still visible.

If a fitted source is inside or adjacent to a newly excluded fit mask, explicitly remove_absorption_source or update it away from the masked/blend region; otherwise the next refit may keep fitting the artifact with a nearby inherited seed.

Adjacency matters visually: a source whose center is just outside a purple exclude band can still be the same masked edge contaminant if its trough or fitted wing touches the mask boundary. In that case, do not treat the unmasked center as independent evidence; remove/update it or make a mask-only round and inspect the next plot.

Never pair a new edge mask with multiple new sources in the partner transition frame unless the current overview image shows multiple independent absorption troughs at distinct observed wavelengths.

Do not mask or alter context-only regions just because they look imperfect if they do not visibly contaminate the source work window or sibling/blend interpretation.

Boundary-band troughs that touch the work window should be handled when the core is already addressed or the edge contaminant biases the fit: either add a boundary source if they look part of the target complex, or apply one narrow mask covering the full contaminant if they look like edge contamination.

Use masks sparingly. A fit mask should tightly cover only the contaminating trough plus a small margin; do not mask broad regions of otherwise usable continuum. Prefer one narrow mask per isolated contaminant, usually under 60 km/s unless the visible contaminant itself is broader. Avoid masks wider than 100 km/s unless the masked feature is a clear projected sibling-transition/blend feature or a saturated contaminant; if so, say that in the reason and expect human review.

## Tool Use

Use one round efficiently, but avoid bundling unrelated guesses. As a default, make 1-3 concrete edits before one request_refit. Overlap bookkeeping is not an unrelated guess: if one edit touches an observed-wavelength overlap, you may exceed 3 edits to address the affected frames with matching masks/windows or to remove/update the source that would otherwise keep fitting the overlap. Use more than 3 source/mask/window edits only for such coupled overlap work or when there are clearly separated high-confidence issues inside the source work window.

In the rationale, summarize the visible evidence and priority order for the edits you chose. Explicitly state why each far-edge mask is safer than adding a source, and whether any high residual in the main absorption complex remains intentionally unhandled this round.

Use fit_control_context.diagnostic_hints: prioritize residual_hints with kind='source_work_window_residual' for source add/update decisions only after checking the wavelength overview. Use add_absorption_source for a distinct missing target trough or shoulder; use update_absorption_source only when moving/tightening one existing component.

Do not update the same component into several different centers in one round. Do not remove or effectively collapse several fitted components unless you call remove_absorption_source with specific evidence for each removal.

Treat residual_hints with kind='source_boundary_residual' as explicit edge decisions: do not ignore them and do not half-mask only the outside part if the absorption visibly starts inside the boundary band.

Treat residual_hints with kind='high_sigma_context_mask_candidate' as red residuals that require a narrow set_fit_mask_interval or an explicit human-review rationale; do not leave them unmentioned just because they are outside the source work window.

Treat weaker context-only residuals outside the source work window as secondary; mask them only when they are narrow, isolated, and likely to contaminate the work-window fit.

If a previous tool call had little effect or was kept out of the main state, choose a materially different edit rather than repeating it.

For a broad trough spanning the wavelength interval between doublet transition markers, consider exploratory add_absorption_source calls on the inner wings or midpoint/blend region in the appropriate transition velocity frames, and consider set_fit_window with sibling_mask_mode='allow_overlap' for both doublet members before refit.

Do not collapse different transitions onto one shared velocity axis; each transition frame keeps its own atomic constants. Treat doublet/multiplet relations as probabilistic context: sibling-supported features are stronger evidence, but outliers should be retained and reported unless there is specific visible evidence for contamination or a fit-control edit.

Do not call request_refit by itself; pair it with a concrete edit.

The loop has an initial experiment budget. After you have seen refit feedback, you may call request_more_budget when one more concrete experiment is justified by the latest result. request_more_budget is not a refit request; it only asks the harness to allow another decision round. Do not pair request_more_budget with source, mask, window, or continuum edits in the same assessment response; make those edits in the next decision round. The harness may approve extra rounds up to its hard cap.

## Structured Context

{{PROMPT_PAYLOAD_JSON}}
