---
name: evaluate-classifier
description: Evaluate a trained ML model and generate a performance report. Use when asked to evaluate, assess, or report on model performance.
argument-hint: [model path or description]
---

When evaluating a trained model, follow these steps:

1. **Load the trained model** (check `output/` directory for joblib or pickle files)
2. **Load test data** and generate predictions
3. **Determine the task type** (regression or classification) and compute appropriate metrics:
   - Regression: RMSE, MAE, R-squared, MAPE
   - Classification: accuracy, precision, recall, F1-score, confusion matrix
4. **Generate diagnostic plots**:
   - Regression: predicted vs actual scatter, residuals vs predicted
   - Classification: confusion matrix heatmap, ROC curve if applicable
5. **Create a feature importance chart** if the model supports it (tree-based models, linear coefficients)
6. **Write an evaluation report** to `output/evaluation_report.md` with:
   - A metrics summary table
   - Key findings and observations
   - Recommendations for improvement
7. **Save all plots** to the `output/` directory

Use polars for data handling. Log all metrics using the project's logging format.
Follow the coding standards in CLAUDE.md.

If $ARGUMENTS specifies a model path or details, use that. Otherwise, look for models in the `output/` directory.
