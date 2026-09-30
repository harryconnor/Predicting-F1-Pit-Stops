# Predicting F1 Pit Stops
Predicting whether a Formula 1 driver will pit on the next lap, using lap-level telemetry and race context. Built for the Kaggle [Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5) competition.

This is a binary classification problem on about 439,000 training laps. Roughly 20% of laps are followed by a pit stop, and models are scored on ROC AUC.

## Results

Best result: 0.9493 out-of-fold AUC (5-fold stratified cross-validation), from a blend of 0.80 CatBoost and 0.20 LightGBM.

## Pipeline

1. Exploratory analysis: class balance, correlations, and pit rate by driver, race, compound and tyre age.
2. Feature engineering. Each driver's laps are ordered within a race, and the features use only the current and earlier laps:
   - lap sequence: previous-lap tyre life, lap time, position and stint; laps since the last recorded lap; rolling 3-lap pace
   - stint pace: average and best lap time on the current set of tyres
   - tyre context: typical tyre life for the race and compound, and the current tyre life relative to it
   - race context and ratios: laps remaining, tyre life per stint, degradation per lap
3. Modelling: LightGBM and CatBoost trained on the same 5 stratified folds with early stopping, using the competition's train set only.
4. Blending: a weighted average of the CatBoost and LightGBM probabilities, with the weight chosen on OOF AUC. Test predictions are averaged across the fold models and saved as a Kaggle submission.


## How to run

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   jupyter notebook f1_pitstop_report.ipynb
   ```