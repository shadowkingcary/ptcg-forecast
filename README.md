# Pokémon TCG forecasting

This repository reproduces the numerical analyses, six tables, and four figures in *Forecasting Pokémon TCG Match Outcomes Using Deck Archetypes and Player Records*. It compares five probability models on the same matches and information cutoffs, then examines history depth, scoring, calibration, record weights, tuning objectives, and update frequency.

Analysis settings are recorded in `config/analysis_protocol.json`; expected paper results and input checksums are in `tests/fixtures/paper_checks.json`. Only the two frozen unified datasets are stored under `data/`. Downloading, joining, de-identification, auditing, modeling, and reporting scripts are included. 

## Reproduce the results

Use Python 3.12 and the four exact package versions in `requirements.txt`. The verified environment uses Python 3.12.14. Reproducing the paper from its frozen inputs needs no network connection, private cache, or other project files once dependencies are installed. Downloading a new source snapshot uses the separate acquisition command below.

Run these commands from the repository directory. Keeping the environment beside the repository avoids including installed packages in the publication files.

```sh
python3.12 -m venv ../ptcg-forecast-env
source ../ptcg-forecast-env/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py --jobs 2
```

The command verifies input checksums and package versions, runs the current tests, fits the models, performs supporting analyses, regenerates tables and figures, and checks the results against the saved paper reference. It runs 20 dependent jobs. Use `--jobs 1` to reduce simultaneous memory use. Allow several gigabytes of working space for predictions and intermediate results. Numerical fitting time depends on the machine.

By default, working files go to `../ptcg-forecast-work/`, beside the repository. Set `PTCG_WORK_DIR` to an absolute directory outside the repository to choose another location. Each analysis run refits the models and overwrites generated results there, leaving both input tables and the frozen test reference untouched. Logs, commands, source/input hashes, package versions, timing, and job status are written to `work/results/full_october_2026_run/manifest.json`. A failed job stops dependent work; its named `.log` contains the error. The runner uses one numerical-library thread per process and a noninteractive plotting backend.

Useful checks without a full refit:

```sh
python scripts/reproduce.py --check-inputs
python scripts/run_tests.py
python scripts/verify_reproduction.py
```

The last command requires completed outputs. It compares dataset counts and the five models' historical, later-period, and October scores with the frozen reference, checks all six table fragments exactly, and checks the Elo, weight-selection, and tuning-objective results. It also requires the history-depth reconciliation checks, October outcome-exclusion check, and four figure files. Floating-point summaries use an absolute tolerance of `1e-10`; table values must match exactly. Figure hashes are recorded, but cross-platform raster identity is not required.

## Inputs and eligibility

Only these two compressed CSV inputs are used:

| File under `data/unified/` | Contents and use |
|---|---|
| `matches.csv.gz` | 1,085,921 prepared rows through October 10, 2026. The loader excludes every row dated October 1 or later, leaving 1,070,023 historical matches. |
| `matches_oct2026_holdout.csv.gz` | All 48,288 eligible matches from October 1–31, 2026. Used only for frozen holdout forecasts and evaluation. |

Do not concatenate the files directly: the first file contains October rows that overlap the second. `scripts/unified_data.py` enforces the cutoff and rejects out-of-period holdout dates. All analyses use that loader. Input checksums distinguish these exact prepared datasets from subsequent source updates.

The records come from Limitless Play tournament data. Each row is one decisive match with both players and recorded deck-category labels present. Ties, double losses, unresolved outcomes, byes, and missing labels were excluded when preparing the tables. Labels use the platform's deck IDs as the operational archetype categories; they are not exact 60-card lists. The `other` category is retained. The schema is:

| Field | Meaning |
|---|---|
| `match_id`, `tournament_id` | Match and event identifiers used to align forecasts and calculate tournament-weighted scores. |
| `tournament_date` | Recorded event start date and time in UTC; determines the information cutoff. |
| `player_A`, `player_B` | De-identified player surrogates linked consistently across the two inputs. |
| `archetype_A`, `archetype_B` | Recorded deck-category identifiers for the two players. |
| `winner` | `A` or `B`, relative to the row's participant order. |
| `round` | Recorded tournament round. |
| `era` | Prepared rotation-era label, retained in the input schema. It is not a predictor in the five current models. |

The loader derives `month` from the date and `y = 1` when A wins. A/B are row labels, not who took the first turn. The included acquisition and preparation scripts implement the raw-to-unified process described below. The two frozen tables remain the reference for exact paper reproduction. Private source records and the corresponding identifier map are stored outside the repository.

## Download and prepare raw data

The pipeline uses the public Standard tournament listing, pairings, and standings endpoints described in the [Limitless tournament API documentation](https://docs.limitlesstcg.com/developer/tournaments). Tournament endpoints do not require an API key; the client respects response rate limits and retries temporary failures. See the [authentication documentation](https://docs.limitlesstcg.com/developer.html#authentication).

```sh
# Download December 30, 2023 through October 31, 2026, then prepare both periods.
python scripts/data_pipeline.py all

# Resume an interrupted collection using its saved tournament listing.
python scripts/data_pipeline.py all --reuse-listing

# Run preparation alone after collection is complete.
python scripts/data_pipeline.py build
```

The date range is start-inclusive and end-exclusive: `--start 2023-12-30 --end 2026-11-01`. `--split-date 2026-10-01` separates history from October. `--work-dir /absolute/external/path` selects another acquisition workspace. For a common location across acquisition and analysis, set `PTCG_WORK_DIR` before running either command. The scripts reject working directories inside the repository.

| Script | Responsibility |
|---|---|
| `download_tournaments.py` | Paginate the listing to the start date, check order and duplicates, download both endpoints for each event, and save retrieval metadata and checksums. |
| `build_unified.py` | Join pairings to standings by stable platform player ID; apply exclusions; check duplicates; construct match IDs; pseudonymize both periods together; validate and write candidate unified tables. |
| `pipeline_utils.py` | API client, response validation, schema, and cache helpers shared by these steps. |
| `data_pipeline.py` | Run collection, preparation, or both. |
| `audit_raw_reconstruction.py` | Compare archived raw records with the frozen paper tables when their corresponding private identifier map is available. |

The work directory contains:

- `raw/`: tournament metadata, endpoint responses, retrieval provenance, and completion report.
- `private/identifier_map.json`: one persistent map for players and tournaments across both periods. Reuse it for resumed or extended builds. It is private and is not a publication file.
- `rebuilt/unified/`: candidate `matches.csv.gz` and `matches_oct2026_holdout.csv.gz`. These candidates use disjoint periods; the historical candidate contains no October rows.
- `rebuilt/build_audit.json`: counts, exclusions, duplicate checks, retained data anomalies, output hashes, and comparisons with the paper's population counts.
- `results/` and `figures/`: analysis outputs generated by `reproduce.py` from the repository's frozen inputs.

Matches are joined using `standings.player`, not display names. Deck categories use `deck.id`, with the original `deck.name` fallback recorded in the audit. The processor excludes byes, missing participants, ties, double losses, unresolved outcomes, and missing archetypes. Match IDs include tournament, round, both participants, and an available bracket-match label. Exact duplicate records are counted and collapsed; conflicting records or ambiguous standings joins stop the build. The same player receives the same surrogate in history and October. Tournament surrogates preserve the ordering of events with identical timestamps.

The original historical dataset contains four same-player pairings. Preparation retains and flags them to reproduce the frozen eligibility rules; it does not silently change the paper's sample.

To import an existing raw archive without network requests:

```sh
python scripts/data_pipeline.py all --offline \
  --listing-file /absolute/path/tournaments.json \
  --import-cache /absolute/path/raw
```

The listing must cover the requested period, and the cache must contain `pairings/<id>.json` and `standings/<id>.json` for each listed event. Imported files receive checksums and are identified as local imports. Missing files, incomplete pagination, conflicting metadata, and damaged caches fail explicitly.

Fresh API downloads can differ from the paper's source snapshots, including later deck reclassification. A fresh private map also assigns different surrogate values. Candidate builds therefore never replace the two frozen repository inputs, and matching population counts alone does not establish identical records. `reproduce.py` continues to use the frozen inputs and verifies their checksums.

For exact reconstruction with the original private map, the builder also accepts `--identifier-map PATH` and repeated `--preserve-labels-from PATH` arguments. The latter retains already-recorded player/event deck labels from specified unified snapshots and counts every preserved label. It requires the map corresponding to those snapshots. The cached-data audit reproduced all 1,070,023 historical rows and 48,288 October rows; October reconstruction retained the recorded labels for 24 player/event pairs that had been reclassified by the source. These private assets are not required to reproduce the paper's analyses from its included inputs.

`audit_raw_reconstruction.py --help` lists the optional historical and October listing/cache paths. It writes aggregate audit results outside the repository and fails on missing or changed rows after the declared label-preservation step. It does not claim that present-day API responses are identical to the archived snapshots.

## Time periods and selection

| Period | Matches | Tournaments | Role |
|---|---:|---:|---|
| December 30, 2023–September 30, 2026 | 1,070,023 | 6,790 | Available historical records |
| July 2024–September 2026 | 926,811 | 5,836 | Primary evaluation, 27 months |
| January 2025–September 2026 | 757,964 | 4,768 | Overlapping later period for recalibration |
| October 1–31, 2026 | 48,288 | 396 | Frozen holdout evaluation |

Initial fitting uses records before April 1, 2024. Core smoothing and joint-model settings are chosen using April–June 2024 validation, with the initial fit held fixed. These validation months do not enter the reported evaluation scores. Historical forecasts thereafter use expanding history frozen at each month's start. Every October prediction uses only pre-October outcomes, including predictions for October 1–10. Neither models, counts, ratings nor recalibration are updated within October.

Core tuning minimizes match-weighted log loss. The logistic penalty is fixed, and overall records inherit smoothing from the archetype-specific model. The supporting Elo benchmark selects settings using tournament-weighted validation loss and successive month-start snapshots. Joint recalibration is fitted on July–December 2024 forecasts and then frozen for the later period and October.

A sensitivity check scores the original candidate grids under both match-weighted and tournament-weighted validation loss. Both objectives select pair smoothing 50, player smoothing `(20, 50)`, and joint scales `(0.28, 0.45, 0.14)`. The fixed logistic baseline and inherited overall-record settings therefore also remain unchanged. No later-period forecasts require refitting; saved forecast checksums and the frozen paper reference verify the reused results. Agreement is limited to the tested candidate sets and does not establish globally optimal settings.

To run this check alone after the main forecast files and paper ledger exist:

```sh
python scripts/tuning_objective_sensitivity.py
```

All 29 candidate scores, selections, evaluation scores, comparisons, and provenance are included in `results/tuning_objective/`. Regenerated outputs and per-match validation audit arrays go to `work/results/tuning_objective/`. Add `--publish-results` to refresh the compact repository bundle after a completed check. Selection is saved before later-period forecast files are read. If a future run selects different settings, the script stops before reusing baseline scores; those models would require new evaluation forecasts.

## Models and supporting experiments

| Paper model | Script | Settings |
|---|---|---|
| Logistic baseline | `scripts/logistic_baseline.py` | Separate focal/opponent archetype indicators, L2 penalty, `C=1`, intercept included. |
| Archetype-pair | `scripts/archetype_pair.py` | Pair win rates shrunk toward 0.5; validation selects pseudo-count 50. Same-label matches receive 0.5. |
| Overall records | `scripts/playerskill_general_only.py` | Pair forecast plus difference in overall-record log odds; overall pseudo-count 20. The original implementation uses `k_s=1e9` to approximate the overall-only limit. |
| Archetype-specific records | `scripts/player_records.py` | Current-archetype records smoothed toward overall records; validation selects `(k_o, k_s)=(20,50)`. |
| Joint model | `scripts/joint_hierarchical_model.py` | Simultaneous archetype, pair, player and player–archetype effects; antisymmetric design, no intercept. Selected block scales `(1,0.28,0.45,0.14)`, `C=1`, tolerance `1e-8`; reaching the 10,000-iteration cap fails. |

The joint search evaluates seven settings: a base configuration and one-at-a-time halving/doubling of its three non-archetype scales. Selected settings remain fixed during evaluation. The mathematical fitting routines and probability-rounding conventions are retained from the paper analysis.

Supporting analyses are limited to those used in the current paper:

- `rating_benchmarks.py`: Elo plus the same pair estimate, with validation-selected update factor 16 and coefficient 1. Ratings start at 1500, update after historical tournaments, and are read from month-start snapshots.
- `posthoc_calibration.py`: frozen two-parameter logistic recalibration of the joint model.
- `rating_fresh_comparison.py` and `card_only_fresh_check.py`: tournament versus monthly updates for archetype-specific and pair records, respectively. Evaluation ends in September. Events are ordered by recorded start date and identifier; exact completion order is unavailable.
- `history_depth_analysis.py`: unchanged forecasts grouped by the smaller of the two players' prior match counts: 0, 1–9, 10–49, and 50+. Counts exclude the entire forecast month and are outcome-independent. Additional outputs distinguish one unseen player from two. Unseen means absent from the eligible records, not inexperienced.
- `player_record_weight_sensitivity.py`: adjustment weights 0, 0.5, 0.75, 1, 1.25, and 1.5. April–June validation selects weight 1 before evaluation or loading October. Weights 0 and 1 reconcile with the pair and overall-record models. Smoothing is unchanged.
- `tuning_objective_sensitivity.py`: compares the existing six pair, sixteen player-smoothing, and seven joint-scale candidates under both validation objectives, preserving sequential selection and inherited overall smoothing. All selected settings agree, so existing evaluation forecasts and comparisons are verified and reused.
- `check_ps_fresh_equivalence.py`: verifies that the vectorized count implementation, when updated monthly, reproduces all 926,811 saved archetype-specific forecasts to rounding precision.

`match_data.py` and `score_utils.py` contain shared encodings, counts, and paired scoring utilities. Saved filenames and result keys retain some earlier naming for numerical continuity: `M1` means logistic baseline, `Card-only` means archetype-pair, `Overall` or `Overall-record control` means overall records, and `Player-Skill` means archetype-specific records. That last key is not a claim that the model directly measures skill.

## Scoring and uncertainty

Primary log loss averages match losses within each tournament and then gives tournaments equal weight. Supporting checks use match-weighted log loss and Brier score, ten-bin match-weighted calibration RMSE, and sharpness measured by mean absolute distance from 0.5. Positive comparison gains mean reference loss minus model loss.

Historical intervals use 5,000 moving-block bootstrap draws with seed 20261011 and four-month blocks. October intervals resample the 396 tournaments independently and are indicative: they omit dependence from repeated players and shared fitted models. Intervals are pointwise and omit model-selection uncertainty. The later historical period overlaps the primary period and is not an independent replication.

History-depth comparisons use common resampling draws and preserve group-empty months and tournaments. A group gives equal weight to tournaments containing that group. Consequently, group gains are not additive and do not test whether effects differ between groups or estimate causal effects of experience. Empty-support replicates are recorded.

Historical predictions and losses are saved independently to six decimal places; October predictions use full precision. The weight experiment uses full precision and reconciles the saved baseline rounding. Existing reversed-contrast bootstrap endpoints are retained. Direct positive-gain intervals can differ by one order statistic. `config/analysis_protocol.json` records these conventions; it documents the implementation rather than serving as a runtime configuration file.

## Outputs and paper mapping

In the table below, `work/` denotes the external workspace, normally `../ptcg-forecast-work/`. Only the unified input tables are kept in the repository as match datasets. The protocol under `config/` and compact expected values under `tests/fixtures/` are configuration and test metadata. Generated predictions, logs, per-match audits, tables, and figures stay in the external workspace. The compact tuning-objective result bundle is also included under `results/tuning_objective/` for publication.

| Paper content | Generated artifact |
|---|---|
| Population counts, five-model scores, paired gains, calibration and sharpness | `work/results/full_october_2026_run/paper_results.json` |
| Tables 1–6 | `work/results/full_october_2026_run/manuscript_tables.json`, keyed by table number |
| Figure 1: design | `work/figures/journal_experimental_design.png` |
| Figure 2: main gains | `work/figures/journal_main_gains.png` |
| Figure 3: history depth | `work/figures/journal_history_depth.png` |
| Figure 4: sharpness and calibration | `work/figures/journal_sharpness_calibration.png` |
| History descriptors, group scores, comparisons and shared bootstrap draws | `work/results/history_depth/` |
| Weight selection, full-precision predictions, comparisons and report | `work/results/player_record_weight/` |
| Tuning-objective candidate scores, selections, unchanged evaluation scores and report | `results/tuning_objective/`; regenerated in `work/results/tuning_objective/` |
| Supporting Elo, recalibration and update-frequency results | `work/results/FINAL_rating_*.csv`, `FINAL_posthoc_calibration.csv`, `FINAL_cardonly_fresh_check.csv` |
| Verification result | `work/results/full_october_2026_run/reproduction_verification.json` |

Table 1 gives historical incremental comparisons; Table 2 uses the same references for October; Table 3 covers history depth; Table 4 gives supporting scores; Table 5 covers recalibration; Table 6 compares update schedules. Table and figure source manifests link reported values to saved summaries and code hashes. Table fragments are stored as strings inside JSON, so no additional Markdown files are generated. The weight report is plain `.txt`.

Expected primary tournament-weighted log losses, in model order, are **0.687591, 0.682405, 0.664820, 0.664577, and 0.662062**. October values are **0.685603, 0.676135, 0.660485, 0.661198, and 0.657270**. Overall records provide most of the gain over archetype information. The small archetype-specific increment depends on the scoring choice and is not clearly supported in October; the joint model improves loss further. The reproduction check compares numerical results rather than relying only on this ranking.

This package includes data acquisition and preparation code and regenerates numerical paper artifacts from the frozen inputs. Manuscript text, journal typesetting, and PDF compilation are outside its scope. Reconstructing exact private identifiers requires the original external map; newly collected snapshots and newly assigned IDs are kept separate from the frozen publication data.
