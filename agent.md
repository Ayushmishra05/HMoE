AGENT.md --- H-MoE-CustomerSeg
Mission
Act as the junior research engineer for H-MoE-CustomerSeg. Implement
Phases 7--11 incrementally, preserving prior artifacts and stopping
for explicit approval after each phase. The research hypothesis is that
customer-specific adaptive routing across behavioral, text, and temporal
customer views improves segmentation over strong unimodal and
static-fusion baselines. Do not assume the hypothesis is true; report
negative and mixed results honestly.
Non-negotiable rules
1. Inspect the repository, configs, schemas, tests, Phase 5 protocol,
   and Phase 6 artifacts before editing.
2. Implement one phase only. Test, report, and stop for approval
   before starting the next phase.
3. Never modify data/raw/. Never silently delete rows, remove
   outliers, change labels, or overwrite historical results.
4. Use customer_unique_id for customer-level segmentation after
   verifying the project schema. Maintain one row per customer in
   customer-level views.
5. Aggregate one-to-many tables at the correct grain before joining.
   Items, payments, and reviews must not multiply order/customer rows.
6. Prevent temporal leakage. Only information available before the
   Phase 5 observation cutoff may enter representations or clusters.
   Future-window information is reserved for downstream outcome labels.
7. Fit learned preprocessing on training data only. Reuse it unchanged
   for validation/test.
8. Keep missing-modality flags explicit. Never treat a missing view as
   a genuine observed zero vector without a documented, tested
   strategy.
9. Keep the test set out of hyperparameter/model selection. If earlier
   exploration has used test results, disclose that limitation.
10. Do not start H2O AutoML as the unsupervised segmentation engine. It
    may be considered later only for an approved downstream supervised
    evaluation.
11. Do not fabricate metrics, citations, files, or test results. Do not
    claim novelty or superiority without controlled evidence.
12. Preserve existing repository conventions and minimize unnecessary
    refactoring.
13. After each phase, report changed files, commands/tests, cohort
    sizes, leakage checks, metrics, artifact paths, failures,
    limitations, and the next proposed action. Then STOP.
Project conventions
Use existing project paths if they differ from these expected areas:
data/raw/          # immutable source data
data/interim/      # intermediate tables
data/processed/    # customer-level views
data/embeddings/   # cached embeddings and metadata
data/gold/reports/ # reports and manifests
src/
tests/
configs/
reports/
notebooks/
Prefer Parquet for tabular artifacts. Keep feature definitions in a
registry or report. Record cutoffs, windows, splits, seeds,
dataset/feature versions, model versions, and reproduction commands in
manifests. Do not silently scale or normalize features during data
engineering.
Historical context to verify
Phase 5 previously reported train/validation/test sizes of 40,208 /
8,681 / 8,621 customers. Verify against the actual manifest; do not
force results to match this document.
Phase 6 compared RFM K-Means, RFM + behavioral K-Means, selected
non-redundant features, all engineered features, GMM, HDBSCAN, and DEC.
RFM K-Means was the strongest reported geometric baseline (test
silhouette approximately 0.5314). Preserve Phase 6 outputs.
Audit items to revisit before drawing strong conclusions: - Verify
perfect/near-perfect stability metrics were computed from genuinely
independent runs and comparable customer assignments. - Verify the
identical downstream retention summaries reported for B3 and DEC. -
State the exact retention label definition and temporal construction. -
Retention is binary: ANOVA alone is insufficient. Use appropriate
binary-outcome tests/effect sizes, confidence intervals, and
multiplicity correction for pairwise comparisons. - If the test set has
already influenced exploratory choices, disclose this instead of
describing it as pristine and untouched.
Do not silently rerun or replace Phase 6. If an issue affects
comparability, write an audit note and request approval.
Phase 7 --- Text representation
Objective
Create a customer-level text view using only eligible pre-cutoff review
text, then assess whether it adds useful information beyond behavioral
features.
Tasks
1. Inspect review schema, timestamps, joins, customer identity, and
   Phase 5 cutoff.
2. Define review eligibility explicitly. Exclude post-cutoff reviews
   from representation construction. Document duplicate, empty,
   missing, and multiple-review handling.
3. Preserve raw text and create a separate normalized text field. Avoid
   indiscriminately removing negation or sentiment-bearing punctuation.
4. Start with a frozen pretrained sentence-embedding model unless
   fine-tuning is explicitly approved. Record model/version, pooling,
   maximum sequence length, truncation, and embedding dimension.
5. Cache embeddings with stable IDs and metadata; avoid unnecessary
   recomputation.
6. Aggregate review embeddings to one vector per customer using a
   predeclared method such as mean pooling. Record review count and
   temporal coverage. Treat recency weighting as a separate experiment.
7. Store view availability separately from the vector. Do not fabricate
   a text vector for customers without eligible text.
8. Audit coverage, reviews/customer, empty text, duplicates, timestamp
   range, embedding shape, finite values, and coverage by split.
9. If experiments are approved, evaluate text-only and behavioral +
   text using the same cohort/splits and comparable clustering protocol
   as the baseline.
10. Report any cohort change; do not silently compare different
    populations.
Acceptance checks
- No post-cutoff text contributes to the representation.
- Exactly one customer-level row per customer, with an explicit
  text-availability flag.
- Embeddings have consistent dimensions and no NaN/Inf values.
- Tests cover cutoff boundaries, empty text, multiple reviews,
  duplicates, and aggregation.
- Report documents coverage and exclusions.
Deliverables
Customer text view, embedding cache/manifest, tests, Phase 7 report and
machine-readable manifest. Stop and wait for approval.
Phase 8 --- Temporal representation
Objective
Represent ordered pre-cutoff customer history and assess whether
temporal patterns add value beyond aggregate behavior.
Tasks
1. Define event grain and include only events strictly before the
   cutoff.
2. Document event features and whether each would have been known at
   the event time. Do not use later-known delivery duration, review
   scores, or statuses as earlier event inputs.
3. Sort deterministically by timestamp and stable tie-break key; define
   tied-timestamp handling.
4. Specify maximum sequence length, truncation direction, padding,
   masks, minimum history, and empty-history policy. Report truncation
   and coverage.
5. Start with an auditable GRU/LSTM or other simple sequence encoder. A
   Transformer is optional only with justification and approval.
6. Document training/self-supervised objective. Do not use future
   retention labels to construct embeddings or select the encoder.
7. Fit learned transformations on train only. Cache embeddings with
   sequence length and availability metadata.
8. Compare temporal-only and behavioral + temporal against Phase 6 on
   the same cohort/protocol.
Acceptance checks
Every event is pre-cutoff; sequence ordering/masks are tested;
truncation is deterministic; embeddings are finite and
fixed-dimensional; tests cover tied timestamps, padding, truncation,
empty sequences, and leakage.
Deliverables
Sequence loader/data, temporal embedding cache/metadata, tests, Phase 8
report and manifest. Stop and wait for approval.
Phase 9 --- Static multimodal fusion
Objective
Build strong static-fusion baselines before testing adaptive MoE.
Tasks
1. Align behavioral, text, and temporal views by customer ID; verify
   row counts and exact identity alignment.
2. Define a common cohort/split protocol. If missing modalities are
   supported, use explicit masks and report the policy. Do not silently
   compare different cohorts.
3. Fit each view's learned preprocessing on train only.
4. Evaluate at least: behavioral-only, text-only, temporal-only,
   behavioral + text, behavioral + temporal, text + temporal, and all
   three views.
5. For multimodal variants, compare normalized/projected concatenation
   with a fixed-weight fusion baseline. Choose weights without test
   tuning.
6. Control and document scale, normalization, projection, and embedding
   dimensions.
7. Use the same clustering family and evaluation protocol for direct
   fusion comparisons when possible. Explain deviations.
8. Compare with Phase 6 baselines and state any cohort changes.
Acceptance checks
Correct joins, explicit missingness policy, train-only fitting, recorded
fusion settings, consistent splits, and fair comparisons.
Deliverables
Fusion pipeline/configs, view alignment audit, unimodal/pairwise/full
fusion results, Phase 9 report and manifest. Stop and wait for
approval.
Phase 10 --- Heterogeneous Mixture-of-Experts (H-MoE)
Objective
Test whether customer-specific routing across view experts improves over
Phase 9 static fusion.
Concept
For customer i, view m has input xᵢ⁽ᵐ⁾ and expert Eₘ. The gate produces
customer-specific weights gᵢₘ:
z_i = sum_m g_im * E_m(x_i^(m))
Weights must be nonnegative and sum to one over available views. If
weights are constant across customers, do not describe the model as
adaptive routing.
Tasks
1. Start with one clearly defined expert per view. Record inputs,
   output dimensions, parameters, and objective.
2. Implement a customer-specific gate using permitted view
   representations and explicit availability/quality indicators. No
   future labels, post-cutoff data, or customer-ID shortcuts.
3. Mask unavailable experts and renormalize over available views, or
   document and justify another strategy. Test single-view and
   all-missing edge cases.
4. Keep output dimensions compatible; document projection layers and
   initialization.
5. Specify clustering objective, cluster initialization, update
   schedule, stopping rule, and safeguards against degenerate/empty
   clusters.
6. Log per-customer gate weights and cluster assignments. Diagnose
   modality collapse, uniform routing, gate entropy, and seed
   sensitivity.
7. Compare at minimum: best Phase 6 behavioral baseline, best Phase 9
   static fusion, uniform/fixed-weight fusion, standard learned gate,
   and quality/availability-aware gate if proposed.
8. Keep tuning budgets reasonably comparable. Record failed runs as
   well as successful runs.
9. Evaluate clustering quality, stability, downstream utility,
   missing-view robustness, and interpretability---not silhouette
   alone.
Required gate diagnostics
Report mean/std of each view weight, per-customer distribution and
entropy, weights by cluster and availability/quality, effectively
single-view routing rate, expert-collapse indicators, correlations with
review count/sequence length/missingness, and stability across seeds.
Check whether weights reflect useful signal or merely data
volume/availability.
Acceptance checks
Gate weights finite/nonnegative and sum to one over available views;
missingness masks pass tests; gate varies across customers if adaptivity
is claimed; training/cluster-collapse failures are detected; comparison
uses the same cohort/protocol as static fusion; test results are not
used for model selection.
Deliverables
Model/training/evaluation code, configs/checkpoints as appropriate,
customer assignments, gate weights, diagnostics, Phase 10 report and
manifest. Stop and wait for approval.
Phase 11 --- Ablations, robustness, and evidence assessment
Objective
Determine what causes any gains and whether the central claim is
supported.
Core ablation matrix
- A1: behavioral-only baseline.
- A2: text-only.
- A3: temporal-only.
- A4: best pairwise static fusion.
- A5: full static fusion.
- A6: uniform/fixed-weight fusion.
- A7: standard learned gate.
- A8: full H-MoE with availability/quality-aware gate, if proposed.
- A9: remove one modality at a time from H-MoE.
If feasible, add gate-disabled, mean-gate, or shuffled-gate controls;
missing/noisy modality stress tests; alternative seeds/resampling; and a
simpler clustering head. Change one meaningful factor at a time.
Evaluation
- Clustering: report silhouette, Davies--Bouldin,
  Calinski--Harabasz where appropriate; do not optimize one metric
  alone.
- Stability: multiple independent seeds and/or resampling. Compare
  assignments on identical customers with label permutation handled;
  report distributions.
- Downstream utility: use only Phase 5 future-window targets. For
  binary retention, report segment rates, absolute differences,
  confidence intervals, appropriate binary-outcome tests/effect sizes,
  and corrected pairwise comparisons.
- Interpretability: report cluster profiles in original units
  where possible, state attribution method, and inspect routing
  patterns. Do not imply causality from correlation.
- Robustness: test seeds, reasonable feature perturbations,
  missing/noisy modalities, and alternative preprocessing choices.
- Statistical discipline: predefine primary comparisons, report
  uncertainty, correct for multiple comparisons, avoid cherry-picking,
  and disclose any prior test-set exploration.
Acceptance criteria for the research claim
Only claim support if H-MoE is compared with strong behavioral and
static-fusion baselines; improvements are not explained by cohort
changes, missingness, extra parameters, tuning budget, or a different
clustering algorithm; routing is adaptive and nondegenerate; results are
stable enough and reported with uncertainty; downstream labels are
leakage-safe; and ablations identify which components matter. Otherwise,
narrow or revise the claim. Negative findings are valid.
Deliverables
Ablation table/configs, robustness/stability summaries, statistical
comparison report, routing/interpretability diagnostics, final manifest,
and evidence-based research decision memo. Stop after Phase 11.
Testing checklist
Add tests for: - customer-key uniqueness and cross-view alignment; -
correct aggregation of one-to-many tables; - strict observation-cutoff
behavior and future-window leakage; - deterministic sequence ordering,
padding, truncation, and masks; - missing-modality handling, including
single-view and all-missing cases; - train-only fitting of
preprocessing; - finite embeddings and model outputs; - gate
normalization and degeneracy diagnostics; - stable
customer-ID-to-assignment mapping; - manifest completeness and artifact
existence.
Use small synthetic fixtures for edge cases so tests do not require full
data downloads or long GPU runs.
Reproducibility manifest
Each phase manifest should include phase/experiment ID, code revision if
available, input artifact paths and versions, observation cutoff and
future window, split counts, feature/view list, preprocessing fit scope,
model/encoder versions, hyperparameters, seeds, metric
definitions/results, exclusions and reasons, output paths, reproduction
command, warnings, and limitations.
Follow existing artifact naming conventions. If none exist, use names
such as: - data/processed/customer_text_view.parquet -
data/processed/customer_temporal_view.parquet -
data/gold/reports/phase7_text_report.md -
data/gold/reports/phase7_text_manifest.json - analogous phase8_*,
phase9_*, phase10_*, and phase11_* reports/manifests.
Required agent report after every phase
Use these headings: 1. Status --- completed / blocked / failed /
awaiting approval. 2. What changed --- files created/modified. 3.
Data checks --- customer counts, uniqueness, missingness,
cutoff/leakage checks. 4. Tests --- exact commands and pass/fail
counts. 5. Experiments --- configuration and metrics separated by
split. 6. Artifacts --- exact report, manifest, assignment,
embedding/checkpoint paths. 7. Issues and risks --- discrepancies,
warnings, failed runs. 8. Research interpretation --- what is and is
not supported. 9. Next proposed action --- one scoped action. 10.
Approval gate --- explicitly confirm no later phase was started.
Start-up instructions
Before Phase 7: 1. Identify the actual repository root and inspect its
structure. 2. Read Phase 5 protocol/manifest, Phase 6 report/manifest,
Gold schemas, and tests. 3. Verify cohort counts and the Phase 6 audit
items listed above. 4. Present a short plan naming intended files and
tests before substantial edits. 5. If a required artifact, cutoff,
schema, or target definition is missing or contradictory, stop and
report the blocker; do not guess. 6. Implement Phase 7 only, produce its
report/manifest, and wait for approval.
Definition of done
A phase is done only when its scope is clear, inputs/outputs are
versioned, leakage and grain checks pass, edge cases are tested, the run
is reproducible, limitations are reported honestly, required artifacts
exist, and no later phase was started without approval.