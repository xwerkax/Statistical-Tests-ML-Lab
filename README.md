# Lab 5 – Statistical Tests for Classifier Evaluation

This laboratory focuses on statistical analysis of classification results. It consists of two main tasks: comparing classifier performance and conducting statistical tests to determine the significance of differences between algorithms.


# Task 1: Classifier Comparison

## Objective
Compare the performance of three classification algorithms on 19 datasets using stratified cross-validation.

## Input Data

- Archive datasets containing 19 files with classification data
- Each file format: last column contains labels, remaining columns are features

## Algorithms to Compare

- Gaussian Naive Bayes (GaussianNB)
- k-Nearest Neighbors (KNeighborsClassifier)
- Multi-layer Perceptron (MLPClassifier)

## Methodology

- Cross-validation: Stratified, 2 folds, 5 repetitions
- Metric: Accuracy score
- Results structure: number_of_datasets × (folds × repetitions) × number_of_classifiers

# Task 2: Statistical Analysis

## Objective
Perform paired t-test for dependent samples to determine statistical significance of differences between classifiers.

## Input Data

- Results file from Task 1 (.npy format)
- Select dataset with the largest number of objects

## Statistical Analysis

- Test: Paired t-test for dependent samples (scipy.stats.ttest_rel)
- Significance level: α = 5% (0.05)

## Matrices to Create
1. t_stat (size: classifiers × classifiers)

- Stores t-statistic values for each classifier pair

2. p_val (size: classifiers × classifiers)

- Stores p-values for each classifier pair

3. better (size: classifiers × classifiers, type: bool)

- Indicates which classifier performed better (not necessarily significantly)

4. stat_significant (size: classifiers × classifiers, type: bool)

- Indicates whether the difference is statistically significant (p < α)

5. stat_better (size: classifiers × classifiers, type: bool)

- Combines better and stat_significant matrices (element-wise multiplication)
- Shows statistically significant superiority
