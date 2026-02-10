---
name: analyze-wine-data
description: Perform exploratory data analysis on the winde dataset from sklearn. Use when asked to explore, profile, or analyze data.
argument-hint: [dataset or file path]
---

When performing exploratory data analysis, follow these steps:

1. **Load the data** into a polars DataFrame. Identify the target variable and feature columns.
2. **Compute summary statistics** including mean, median, std, min, max for each numeric feature
3. **Check for missing values** and report the count and percentage per column
4. **Check for duplicate rows** and report how many exist
5. **Generate distribution plots** for each numeric feature using matplotlib histograms
6. **Create a correlation matrix** heatmap using seaborn
7. **Identify outliers** using the IQR method and log the count per feature
8. **Log a summary** of key findings using the project's logging format
9. **Save all plots** to the `output/` directory

Use polars (not pandas) for all data manipulation. Follow the coding standards in CLAUDE.md.

If $ARGUMENTS specifies a dataset or file path, use that. Otherwise, ask the user what data to analyze.
