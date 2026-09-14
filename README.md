# Carbon risk screening for ready-mixed concrete EPDs

Research code for ranking ready-mixed concrete Environmental Product Declarations (EPDs) by their risk of high strength-normalized embodied carbon, using only metadata available before award.

The screening target is A1–A3 global warming potential per 1,000 psi of declared compressive strength. A declaration is labeled high carbon when that quantity falls in the top decile of the modeling sample. Every environmental outcome field is excluded from the predictor set, so the score depends only on information a procurement team holds before a mix-specific life cycle assessment exists.

---

## Data

The analysis uses a public compiled dataset of United States ready-mixed concrete EPDs. It is **not redistributed here**; download it from the source:

> Broyles, J. (2026). *Compiled dataset of concrete mixture environmental product declarations in the U.S.A.*, Version 5 [Data set]. Mendeley Data. https://doi.org/10.17632/r4jgxk2mhn.5

Place the CSV at `data/concrete_epd_mendeley_v5.csv`, then point the pipeline at it:

```bash
export CONCRETE_EPD_CSV=data/concrete_epd_mendeley_v5.csv     # Windows: set CONCRETE_EPD_CSV=...
```

Every script reads this variable. Without it the pipeline falls back to an earlier dataset version and results will not match.

After filtering to records with positive GWP and positive declared strength, and trimming above the 99.5th percentile of GWP and of carbon intensity, the modeling sample is **46,917 declarations** from 83 producers, 640 plants and 82 source links, issued 2021–2025.

## Environment

```bash
python -m pip install -r requirements.txt
```

Pinned versions used to produce the released results:

| Package | Version |
|---|---|
| Python | 3.13.14 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| LightGBM | 4.6.0 |
| XGBoost | 3.2.0 |

Gradient boosting runs on GPU where available and falls back to CPU automatically. GPU and CPU runs agree to about ±0.005 ROC AUC.

---

## Reproducing the results

Run in this order. Each script writes to `results/` and prints a manifest recording its inputs and parameters.

```bash
cd src

# 1. the reported model on the three holdouts
python run_acrm_model.py --learner lgbm --disable-priors

# 2. benchmark against thirteen alternative learners, then the paired
#    bootstrap of the reported model against the XGBoost baseline
python run_concrete_epd_algorithm_benchmark.py
python run_concrete_epd_v5_validation.py
python run_paired_bootstrap.py

# 3. comparable-group opportunity and its intervals
python run_concrete_epd_procurement_opportunity.py
python run_opportunity_intervals.py

# 4. robustness suite: repeated partitions, ablations, alternative targets,
#    duplicate handling, label threshold, held-out operator and region
python run_validation_suite.py

# 5. decision-analytic evaluation and practice baselines
python export_sample_keys.py
python run_decision_analysis.py
python run_ceiling_comparison.py

# 6. further checks
python run_hyperparameter_search.py
python run_feature_contribution.py
python run_compound_shift.py
python run_ranking_objective.py
python run_regression_formulation.py

# 7. external corpora: retrieve from the open registries, map, and score the frozen model
python fetch_external_registries.py environdec
python fetch_external_registries.py epdnorge
python extract_environdec.py
python extract_epdnorge.py
set EXTERNAL_TAG=australia && set EXTERNAL_CSV=../data/external/environdec_ready_mixed.csv
python run_external_validation.py && python run_external_checks.py && python run_external_regime.py && python run_external_updating.py
set EXTERNAL_TAG=nordic && set EXTERNAL_CSV=../data/external/epdnorge_ready_mixed.csv
python run_external_validation.py && python run_external_checks.py && python run_external_regime.py && python run_external_updating.py

# 8. figures
python make_figures.py
python make_decision_figures.py
```

Two large intermediate files are **not** shipped because they are regenerable and exceed a reasonable repository size: the whole-sample key table (`export_sample_keys.py`) and the enriched prediction export (`run_validation_suite.py`, baseline task). Run those two scripts before the analyses that consume them.

---

## Where each result comes from

| Reported item | Script | Result file |
|---|---|---|
| Sample profile | `concrete_epd_pipeline.py` | `dataset_summary.csv` |
| Model performance, three holdouts | `run_acrm_model.py` | `acrm_single_model_results.csv` |
| Benchmark of fourteen learners | `run_concrete_epd_algorithm_benchmark.py` | `final_all_algorithm_metrics_with_acrm.csv` |
| Paired bootstrap against the boosted baseline | `run_paired_bootstrap.py` | `paired_bootstrap_acrm.csv` |
| Top-k screening, calibration | `run_acrm_model.py` | `acrm_single_model_operational_metrics.csv` |
| Comparable-group opportunity | `run_concrete_epd_procurement_opportunity.py` | `procurement_opportunity_*_summary.csv` |
| Opportunity and capture intervals, clustered | `run_opportunity_intervals.py` | `opportunity_point_estimates.csv`, `opportunity_clustered_ci.csv` |
| Interval width by resampling scheme | `run_decision_analysis.py` | `clustered_bootstrap_ci.csv`, `bootstrap_ci_width_comparison.csv` |
| Test partition composition | `export_sample_keys.py` | `split_profile.csv` |
| Stability across repeated partitions | `run_validation_suite.py` | `stability_replicates.csv`, `stability_summary.csv` |
| Feature ablations | `run_validation_suite.py` | `ablation_metrics.csv` |
| Discrimination within strength class | `run_decision_analysis.py` | `within_strength_class.csv` |
| Alternative target normalizations | `run_validation_suite.py` | `target_sensitivity.csv` |
| Label threshold from training rows only | `run_validation_suite.py` | `train_only_threshold.csv` |
| Duplicate audit and de-duplicated fit | `run_validation_suite.py` | `duplicate_audit.csv`, `duplicate_sensitivity.csv` |
| Held-out program operator and region | `run_validation_suite.py` | `quasi_external_validation.csv` |
| Compound producer and period shift | `run_compound_shift.py` | `compound_shift.csv`, `compound_shift_summary.csv` |
| Net benefit and cost-sensitive budget | `run_decision_analysis.py` | `decision_curve.csv`, `cost_sensitive_optima.csv` |
| Practice baselines and achievable ceiling | `run_decision_analysis.py`, `run_ceiling_comparison.py` | `practice_heuristic_baselines.csv`, `ceiling_vs_practice.csv` |
| Hyperparameter search | `run_hyperparameter_search.py` | `hpsearch_configurations.csv`, `hpsearch_manifest.json` |
| Contribution of the feature construction | `run_feature_contribution.py` | `feature_contribution_summary.csv` |
| Ranking objective comparison | `run_ranking_objective.py` | `ranking_objective_inner.csv`, `ranking_objective_holdout.csv` |
| Regression formulation comparison | `run_regression_formulation.py` | `regression_formulation_inner.csv`, `regression_formulation_holdout.csv` |
| External corpora, retrieval and mapping | `fetch_external_registries.py`, `extract_environdec.py`, `extract_epdnorge.py` | `data/external/*_ready_mixed.csv`, `external_*_manifest.json` |
| Transfer to independently compiled corpora | `run_external_validation.py` | `external_australia_validation.csv`, `external_nordic_validation.csv` |
| Learner, mapping, and learnability checks | `run_external_checks.py` | `external_*_checks.csv` |
| Regime-matched controls and regression score | `run_external_regime.py` | `external_*_regime_controls.csv` |
| Model updating with local data | `run_external_updating.py` | `external_*_updating.csv` |

Figures are in `figures/`.

---

## Method notes

**Leakage control.** Twenty-five environmental outcome columns are dropped from the predictor set: every A1–A3 GWP component, the other LCA midpoints, waste and freshwater outcomes, and the derived carbon-intensity target. Retained predictors are supplier and plant identity and location, compressive strength and curing, mixture composition flags parsed from component text, application and specification descriptors, and TF-IDF features of non-outcome free text.

**Validation.** Three holdouts, each reserving 20% of records: unseen producer, unseen EPD source, and a temporal split on issue date. Grouped designs are repeated over ten partitions; the temporal design over four rolling-origin windows.

**Preprocessing hygiene.** The TF-IDF vocabulary, categorical encodings, imputation values and scaling statistics are fitted on training rows only and applied unchanged to the test partition. The operating threshold is fixed at 0.5 and never selected on test data. The carbon-intensity percentile defining the label is computed on the full modeling sample; `train_only_threshold.csv` reports the effect of estimating it from training rows instead.

**One set of scores.** `run_acrm_model.py` writes the reported model's out-of-sample scores to `acrm_single_model_predictions.csv`; every downstream analysis (decision curves, practice baselines, within-class discrimination, clustered intervals, paired bootstrap) reads those stored scores rather than refitting, because GPU histogram construction is not bit-reproducible and a refit differs in the third decimal.

**External corpora.** Two corpora independent of the development data are retrieved from open digital EPD registries (EPD International, principally Australian declarations; EPD-Norge, principally Danish and Norwegian) as ILCD JSON, mapped to the predictor schema (cylinder MPa × 145.038 = psi; supplier, plant, region, and operator treated as unseen), and scored by the U.S.-fitted model without refitting. The raw registry records are not redistributed; `fetch_external_registries.py` retrieves them and the mapped corpus tables are in `data/external/`. Australian producer mix codes carry the strength grade in their leading digits; the rule is validated against the 115 records that state both (98.3% agreement).

**Uncertainty.** Declarations are clustered within producers, plants and sources. Intervals are reported under row-level resampling and under cluster resampling by producer, plant and source, with issue quarter as a block for the temporal holdout.

---

## Citation

If you use this code, please cite the dataset above. Citation details for the accompanying work will be added here once available.

## License

Released under the MIT License; see `LICENSE`. The dataset is distributed by Mendeley Data under its own terms.
