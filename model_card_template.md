# Model Card

## Model Details
This model is a **RandomForestClassifier** (scikit-learn, 100 estimators, max_depth=12, random_state=42) trained to predict whether an individual's annual income exceeds $50,000 based on US Census demographic and employment data. Categorical features are one-hot encoded; the target
label is binarized (<=50K -> 0, >50K -> 1).

## Intended Use
This model is intended as an experimental income-based prediction artifact for an MLOps pipeline (training, testing, API deployment, CI/CD). It is not intended for use in any real decision-making process that affects individuals -- e.g. lending, hiring, benefits eligibility, or any context where a prediction about someone's income bracket could influence outcomes for them. The training data and target variable encode historical social and economic patterns, including known disparities by sex, race, and national origin; using this model outside an experimental context risks reproducing those disparities at scale.

## Training Data
Source: [UCI Machine Learning Repository, "Census Income" / "Adult" dataset (extracted from the 1994 US Census database)](https://archive.ics.uci.edu/ml/datasets/census+income).

The raw CSV had a leading whitespace character after every comma (header and values), which was removed via a scripted cleaning step (data_cleaning/data_cleaning.py) prior to use. No other transformation was applied to the raw data; in particular, the ? placeholder used for missing categorical values (e.g. in workclass, occupation, native-country) was retained and treated by the one-hot encoder as its own category, rather than imputed or dropped.

An 80/20 train/test split was used, stratified on the target label, with random_state=42 for reproducibility.

## Evaluation Data
The held-out 20% test split (6,513 rows) from the same source dataset, processed identically to the training data (same cleaning step, same fitted encoder/label binarizer reused in inference mode, no separate preprocessing).

**Metrics**

Overall performance on the test set:

## Metric      Value
Precision:   0.8083
Recall:      0.5727
F1:          0.6704

The model is precision-heavy and recall-weak: when it predicts >50K it is usually correct, but it misses a substantial fraction of actual >50K cases. This trade-off should be made explicit to anyone
considering use of this model -- it is conservative about flagging high income, not balanced.

Slice Performance

Full per-category-value performance is in slice_output.txt. Two findings worth flagging explicitly rather than burying in the raw table:

- Sex: recall is lower for Female (0.5102, n=2158) than for Male (0.5843, n=4355). Both slices have large enough sample sizes that this gap is unlikely to be pure noise. The model is more likely to under-predict high income for women than for men.
- Race: recall varies more widely across race categories (e.g. Black: 0.4839, n=662 vs. White: 0.5773, n=5533), though some of this variation is on smaller samples and should be read with more caution than the sex disparity above.


Many other slice rows (e.g. workclass: Never-worked, n=1, most native-country values outside United-States, n<30) have very small sample sizes. Precision/recall of 1.0 or 0.0 on those rows reflects sample size, not genuine model performance, and should not be interpreted as a finding.

## Ethical Considerations
This dataset encodes protected/sensitive attributes directly (race, sex, native-country) and via proxies (occupation, relationship, marital-status). The slice analysis above shows a measurable recall disparity by sex. This model should not be treated as fairness-neutral. Any downstream use - even illustrative - should disclose this disparity rather than present the overall F1 score as a complete picture of model behavior.

## Caveats and Recommendations
- The ? missing-value category was not imputed; its effect on per-feature slices including it (e.g. workclass: ?) has not been separately characterized beyond the slice numbers shown.
- Small-sample slices in slice_output.txt are noisy and should be read with that caveat, not as reliable per-group performance.
- This model has not been tuned beyond default-ish hyperparameters (100 trees, max_depth=12); no hyperparameter search was performed. Reported metrics reflect a reasonable baseline, not the maximum achievable performance from this dataset/algorithm combination.
- Given the sex-based recall disparity, this model should not be used as-is in any application where under-identifying a group's high-income cases would cause harm.
