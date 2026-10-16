# Goal008 exploratory association definition

Protocol ID: `goal008_cpu_proxy_associations_v1`.

This definition is fixed before calculating the new proxy scores or inspecting
their associations. It specifies a new exploratory analysis, not a retrospective
claim of preregistration and not an extension of Goal007's twenty confirmatory
comparisons. The conventional RQ2 status remains
`NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES`.

## Inputs and binding

The unit is one frozen O description: 60 buildings by three description
providers, 180 descriptions. The two Builders are repeated outcomes of each
description; they are never pooled as 360 independent descriptions. O is the
saved JSON visual-description record, not free-form prose.

Each score row must bind `scene_id`, `description_provider`, `input_id`,
`input_sha256`, `source_sha256`, `gt_bbox_sha256`, and `gt_voxels_sha256` to both
corresponding O outcome rows. Outcome rows retain Goal007 fields and receive the
last three hashes only through the audited source map. No field aliases,
approximate text matches, historical scores, or fallback descriptions are used.
The actual analysis requires a passing Phase 0 binding gate, 180 exact matches,
and this frozen definition's SHA256. Fixtures are separately labeled.

Five fixed features, each in [0,1], are analyzed separately:
`strict_material_f1`, `coarse_material_f1`, `dimension_accuracy`, `completeness`,
and `composite`. The scorer's separately fixed definition determines their
meaning and missing-safe construction. A null feature is excluded only from
analyses using that feature, must have an explicit `missing_reasons` entry, and
is not converted to a favorable value by this module. No weight, normalization,
feature or outcome is chosen after seeing an association.

## Fixed analysis grid

The outcome is aligned occupancy IoU from O, using Goal007's strict metric.
Every combination below is emitted, including nonestimable combinations:

- Five features.
- Four scopes: all description providers, OpenAI, Anthropic, and Gemini.
- Two outcome policies: valid-only and separate bounded-IoU failure0 sensitivity.
- Three targets: OpenAI Builder, Anthropic Builder, and the arithmetic mean of
  both Builders where both outcomes are eligible under the stated policy.
- Three associations: pooled-description Spearman; Pearson of within-building
  centered feature/outcome values; Pearson of within-building midrank-centered
  feature/outcome values, labeled a within-building rank association.

This produces 360 records. Failure0 is permitted only for identified completed,
known-usage parse, validation, execution or truncation failures. Operational
unknowns and other noneligible outcomes remain missing. The two-Builder mean
requires both values; a one-Builder mean is never substituted.

Pooled Spearman gives each complete description pair equal weight. It is not an
equal-building mean, and complete-case selection can change the population.
Building clustering determines uncertainty. For within-building analyses,
centering and midranks use the same complete feature/outcome pairs within each
building. Buildings with fewer than two complete pairs contribute no within-
building observations. Eligible centered description pairs are equally weighted.
Ties receive their average rank. Constant within-building vectors are exactly
zero-centered. The within-building rank statistic is Pearson correlation of
those rank deviations; the deviations are not globally reranked.

A description-provider stratum has only one description per building, so its
within-building associations are explicitly nonestimable. These rows are kept
with their denominators and reason; they are not silently pooled across strata.

## Cluster bootstrap and nonestimability

Seed is **20261016**, generator is NumPy **PCG64**, and replicates are **2000**.
Buildings are sorted by scene ID. Each replicate samples the same number of
building clusters as the planned scope with replacement, retaining every
description from each sampled building before applying the fixed complete-pair
mask. Planned buildings with no eligible pairs remain in the sampling frame.
The same building-draw matrix is shared by all fixed analyses of that frame.

A repeatedly sampled building is a separate cluster copy. Its observations are
duplicated, not collapsed to one building. An exact computational shortcut uses
integer multiplicities: pooled Spearman midranks are recalculated for each
replicate's multiplicities, and within-building centered/ranked vectors are
copied with those multiplicities. This is equivalent to explicitly assigning a
distinct bootstrap cluster ID to each draw and centering each copy separately.

The point coefficient is reported when at least two eligible pairs and variation
in both variables permit it. A cluster interval additionally requires at least
two contributing buildings. The 95% percentile interval uses the 0.025 and 0.975
quantiles with NumPy's linear interpolation. Undefined replicates are counted;
at least 1900 of 2000 (95%) must be finite or the interval is null and labeled
`UNSTABLE_BOOTSTRAP`. A mathematically defined point coefficient is retained even
when its interval is unavailable. Constant proxy/outcome, insufficient pairs,
insufficient buildings and single-description strata have distinct reasons.

For every record, report planned descriptions/buildings, feature/outcome missing
counts and their overlap, complete pairs/buildings, contributing within-building
pairs/buildings, excluded singleton buildings, tied observations/groups, unique
values, constant groups, valid/invalid bootstrap replicate counts, and interval
status. Bootstrap copies never inflate the reported original building N.

No p-values, significance labels, additional Holm corrections, sign-flip tests
or optimized metric selection are produced. The original Goal007 twenty tests
are read-only and are not rerun by this module.

## Interpretation limits

The proxy is a new CPU-derived attribute fidelity proxy, not conventional
caption quality, an independent human judgment, or evidence of causality.
Proxy and reconstruction outcomes share GT; dimension and material associations
can reflect this construction. Cluster intervals address dependence, not proxy
validity, missingness, selection bias or exploratory multiplicity. Within-building
associations remove additive building-level offsets but do not establish a
causal description-quality effect. All fixed analyses and null results remain
visible, and no selection among them resolves conventional RQ2.
