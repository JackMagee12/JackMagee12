# Portfolio revision notes

The original coursework notebook is preserved unchanged in the downloadable package's `original/` folder. The revised notebook and `analysis.py` were prepared with AI assistance for presentation and validation corrections. The revision keeps the project's business question and model families while making the changes below explicit.

## Presentation

- Reorganized question-number answers into a business question, workflow, findings, evaluation, and limitations.
- Generated fresh outputs and removed conflicting manually transcribed scores from the revised version.
- Added a README, dependency versions, input checks, charts, metric exports, and a standalone Python script.
- Replaced unsupported causal and deployment conclusions with qualified interpretations.

## Baseline models

- Fixed random seeds for decision tree and random forest models.
- Kept the original 70/30 stratified split, five-fold stratified cross-validation, F1 objective, and searched parameter values.
- Retained logistic regression's default L2 behavior without the deprecated explicit `penalty` argument and increased its iteration limit to 2,000.
- Added an always-stay baseline and ROC-AUC/average-precision metrics.
- Limited parallel jobs to two to make execution manageable.
- Corrected tenure-bin labels and included zero tenure in the first bin.

## Stacking experiment

- Replaced the unavailable `category_encoders.TargetEncoder` dependency with scikit-learn's cross-fitting `TargetEncoder`. These encoders are not identical, so the old 0.7848 AUC is not a claimed reproduction target.
- Moved preprocessing into each stacking base estimator's pipeline so the encoder is fitted separately within stacking folds.
- Used the baseline models' same stratified 70/30 evaluation split instead of the original separate, non-stratified 80/20 split.
- Kept the original gradient-boosting/random-forest/logistic-regression stacking idea and documented the service-count definition.
- Refit the pre-specified ensemble on all labeled rows only after evaluation to generate predictions for the unlabeled file.

## Scope of validation

The portfolio notebook's cells were executed sequentially in a clean Python process, and the delivered notebook includes those new outputs. Input schemas, prediction row count, and probability bounds were checked. Python package versions are recorded. Jupyter's front end was not used to execute the notebook.

This was a retrospective validation using data already explored in the coursework. No fresh external or prospective dataset was available. Neither the original submission nor the revised experiment establishes a deployment-ready model.
