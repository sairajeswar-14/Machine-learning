# Predicting Crop Diseases Before Visible Symptoms (ML demo)

Working demo of the paper *"Predicting Crop Diseases Before Visible Symptoms Using Machine Learning"*
(Gali Sai Rajeswara Reddy, Sesi Reddy, A Yaswanth Reddy - Presidency University, Bengaluru).

> **All data here is SIMULATED.** As the paper states, if no real pre-symptomatic dataset is available,
> the sensor channel is simulated and reported as simulated. Results show the *pipeline works*,
> not real field performance.

## Run it (2 minutes)
```bash
pip install -r requirements.txt
python src/train.py      # builds data, trains models, prints Table III
python src/run_demo.py   # risk map + alert list + timeline (saved in results/)
```

## The idea in one line
A plant is stressed **days before** a spot is visible. Sensor drift + warm/wet weather tell us early.

## How the code maps to the paper
| File | Paper section | What it does |
|---|---|---|
| `src/simulate.py` | V-A Data | 14 fields x 25 zones x 70 days. Infection follows warm/wet weather; a 4-8 day hidden phase follows, where NDVI falls, canopy-air temperature gap rises, red-edge shifts. Rows stop at the first visible symptom. Includes a *drought* confounder (looks like stress, not disease). |
| `src/features.py` | IV-C Features | Day-to-day change and 3/5/7-day slopes of sensor indices; leaf-wetness hours, rolling humidity, favourable days. |
| `src/models.py` | IV-D Two-stage | **Stage 1** Random Forest: already infected? **Stage 2** boosted model on the last 5 days of features + Stage-1 probability: symptoms within 5 days? (optional real LSTM in `lstm_stage2.py`). |
| `src/evaluate.py` | III-B, V-C | Threshold chosen to minimise J = 10 x misses + 1 x false alarms; recall, precision, F1, AUROC, lead time. |
| `src/train.py` | V-B, V-D | Baselines (symptom-only, weather-only, sensor-only) vs fusion. Split **by field**, not by random rows. |
| `src/run_demo.py` | IV-E Alerts | Zone risk map with watch / warning / act-now levels, ranked list, timeline. |

## Results
Table III is filled by `train.py` into `results/table3.csv`. Typical run (simulated):

| Method | Recall | Precision | F1 | AUROC | Lead time (days) |
|---|---|---|---|---|---|
| Symptom-only (reactive) | 0.00 | 0.00 | 0.00 | 0.50 | 0.0 |
| Weather-only RF | 0.00 | 0.00 | 0.00 | 0.59 | 0.0 |
| Sensor-only RF | 0.70 | 0.33 | 0.45 | 0.94 | 3.9 |
| Proposed fusion | 0.66 | 0.41 | 0.51 | 0.95 | 3.5 |

Honest reading: sensors carry most of the signal; fusion gives better precision/F1 (fewer false alarms)
but not longer lead time in this simulation. Real data may differ.

## Limitations (same as the paper)
Simulated data, label scarcity, drought confusion, sensor cost, domain shift - a field trial is needed.
