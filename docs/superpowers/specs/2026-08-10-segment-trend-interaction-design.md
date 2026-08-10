# Design: `segment` x trend-family interaction features

Date: 2026-08-10
Status: Documented, ready to validate in notebook

## Goal

Try to improve on the tuned LightGBM baseline (combined score
**0.20752 ± 0.00332**, `config.TUNED_LGBM_PARAMS`) via feature
engineering, now that regularization tuning (Step 13, see
`docs/superpowers/specs/2026-07-31-lightgbm-regularization-design.md`)
failed to close the confirmed train/val overfitting gap without giving up
val score. That result showed the gap isn't primarily a "model has too
much capacity for these features" problem -- the next lever is what the
model is fed, not how much it's allowed to fit it.

## Motivation

Investigated 2026-07-31 (quick AUC/target-rate check on the dataset's
customer-profile columns): `segment` (HVC/LVC/MVC) has by far the
strongest untapped categorical signal -- target rate 19.0% (HVC) / 17.2%
(LVC) / 12.6% (MVC) vs. a ~15% base rate. It's already one-hot encoded in
the production pipeline (`features.encode_features()`), but **no
interaction** with the financial trend features exists yet, e.g. does a
`daily_avg_bal` drop mean something different for an HVC customer (High
Value, presumably larger balances/more transactional headroom) vs an MVC
customer (Medium/lower Value)? Other customer-profile columns checked in
the same investigation (`region`, `arpu`, `age`, `gender`, `smartphone`,
`earning_pattern`) showed weak or no signal and are deprioritized (see
`README.md` Progress log, 2026-07-31 entries, and project memory).

## Source

**Zheng & Casari, "Feature Engineering for Machine Learning"** (see
`RESOURCES.md`) -- interaction features across variables. Already cited
in this repo for `features.build_net_flow_features()` (a numeric x
numeric interaction: inflow - outflow). This round applies the same
general technique to a categorical x numeric pair: multiplying a
category's one-hot indicator by a numeric trend feature isolates that
feature's effect within each category, which is exactly the "does X mean
something different depending on Y" question motivating this round.

## Design

1. New `features.build_segment_interaction_features()` in
   `src/features.py`:
   - Input: the raw `segment` column + the already-computed trend
     features (`build_monthly_trend_features()` output, or the df it's
     built from).
   - For each of `config.TOP_TREND_FAMILIES` (13 families), take the
     single stat already validated as the strongest per-family signal in
     Step 2 (`{family}_delta_m1_m6` -- the raw ranking criterion that
     picked these 13 families in the first place) and multiply it by each
     of the 3 `segment` one-hot indicators (HVC/LVC/MVC).
   - Scope note: limited to `delta_m1_m6` (not all 7 stats per family from
     `_trend_stats()`) to keep the added feature count manageable
     (13 families x 3 segments = 39 new columns) -- consistent with this
     project's "one change at a time" discipline. If this round shows a
     clear win, expanding to more stats per family is a natural follow-up,
     not bundled into this round.
   - Implementation: build the 3 `segment` indicators directly from the
     raw `segment` column inside this function (full one-hot, no
     `drop_first`), rather than reusing `features.encode_features()`'s
     output -- that function applies `drop_first=True` (drops one
     reference category to avoid collinearity in the base feature set),
     which would silently reduce this round to 13 x 2 = 26 interaction
     columns and make the interaction depend on `encode_features()`'s
     internal, alphabetically-determined choice of which category to
     drop. Interaction features don't carry the same collinearity
     concern as the base one-hot encoding, so all 3 segments are used.

2. Validate in `notebooks/01_eda_trends.ipynb` (Step 14): add these 39
   columns on top of the current production feature set
   (`features.build_production_features()`), score with
   `evaluate.cross_validate_score()` using the tuned, calibrated LightGBM
   pipeline (`model.build_lightgbm_calibrated_pipeline(class_weight=None,
   **config.TUNED_LGBM_PARAMS)`), and compare against a same-run baseline
   (current production features, no interaction columns) -- not the
   stale README number, per this project's standing rule.

3. Gate (this project's standard rule, step 6 of
   `structuring-ml-projects`): adopt into `features.build_production_features()`
   (and therefore `src/train.py`) only if the interaction features
   improve or hold **both** Log Loss and ROC-AUC vs. the same-run
   baseline. If not, log as a negative result in `README.md`, keep the
   function available but unused -- same treatment as the four prior
   rejected experiments.

4. If adopted, also re-check the train/val gap (reuse the Step 12/13
   fold-loop pattern) as a secondary observation -- not a gate condition
   for this round (this round's goal is combined score, not the gap
   directly), but worth recording since more features could plausibly
   move the gap either direction.

## Out of scope (deferred)

- Interacting `segment` with trend stats beyond `delta_m1_m6` (mean_6m,
  std_6m, ratio_m1_mean, etc.) -- follow-up only if this round wins.
- Interacting `segment` with the remaining 16 lower-priority trend
  families -- these already ranked weakest in the original family
  ranking (notebooks/01_eda_trends.ipynb Cell 4), unlikely to reverse
  that by adding a segment split.
- `region` x trend interactions -- `region` showed weaker standalone
  signal (13.4%-15.5% spread) than `segment` in the 2026-07-31
  investigation; a future round if `segment` interactions win and the
  technique is worth repeating.
- Group-relative features (e.g. z-score of a trend feature within its
  segment, instead of indicator x value) -- a different formulation of
  the same idea, worth trying as a follow-up if the simple product
  interaction under-delivers.

## Status

Implemented and adopted 2026-08-10. Step 14 result: combined score
0.20763 -> 0.20755 (Log Loss 0.27051 -> 0.27036, ROC-AUC held at
0.88668) -- improved/held on both metrics, so wired into
`features.build_production_features()`. Full numbers in `README.md`
Progress log.
