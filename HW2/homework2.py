# -*- coding: utf-8 -*-
"""homework2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tFL8XUFiFuamH3xum2Me6igDAxOB_esg

## Homework 2: Intro to bias and fairness

Download the German Credit dataset: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
(use the “numeric” version of the data)

Implement a (logistic regression) classification pipeline using an 80/20 test split. Use a regularization value of C = 1.

Treat the 20th feature (i.e., feat[19] in the numeric data, which is
related to housing) as the “sensitive attribute” i.e., z=1 if the feature value is 1.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

german_credit = np.loadtxt("german.data-numeric")
attrs = german_credit[:, :-1]
labels = 2 - german_credit[:, -1]

split_point = 800
X_train, X_test = attrs[:split_point], attrs[split_point:]
y_train, y_test = labels[:split_point], labels[split_point:]

sensitive_attribute = 19

model = LogisticRegression(C=1, max_iter=1000, random_state=42)

"""1. Report the prevalence in the test set."""

prevalence = np.mean(y_test)

prevalence

"""2. Report the per-group prevalence for z=0 and z=1."""

z_test = (X_test[:, sensitive_attribute] == 1)

prevalence_0 = np.mean(y_test[(z_test == 0)])
prevalence_1 = np.mean(y_test[(z_test == 1)])

prevalence_0, prevalence_1

"""3. What is the demographic parity (expressed as a ratio between z=0 and z=1) for your classifier on the test set?"""

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

z_test = (X_test[:, sensitive_attribute] == 1)

positive_pred_z_0 = np.mean(y_pred[z_test == 0])
positive_pred_z_1 = np.mean(y_pred[z_test == 1])

parity = positive_pred_z_0 / positive_pred_z_1

parity

"""4. Report TPR_0, TPR_1, FPR_0, and FPR_1 (see "equal opportunity" slides)."""

z_test = (X_test[:, sensitive_attribute] == 1)

tp_0 = np.sum((y_pred == 1) & (y_test == 1) & (z_test == 0))
fn_0 = np.sum((y_pred == 0) & (y_test == 1) & (z_test == 0))
fp_0 = np.sum((y_pred == 1) & (y_test == 0) & (z_test == 0))
tn_0 = np.sum((y_pred == 0) & (y_test == 0) & (z_test == 0))

tp_1 = np.sum((y_pred == 1) & (y_test == 1) & (z_test == 1))
fn_1 = np.sum((y_pred == 0) & (y_test == 1) & (z_test == 1))
fp_1 = np.sum((y_pred == 1) & (y_test == 0) & (z_test == 1))
tn_1 = np.sum((y_pred == 0) & (y_test == 0) & (z_test == 1))

TPR_0 = tp_0 / (tp_0 + fn_0)
TPR_1 = tp_1 / (tp_1 + fn_1)

FPR_0 = fp_0 / (fp_0 + tn_0)
FPR_1 = fp_1 / (fp_1 + tn_1)

TPR_0, TPR_1, FPR_0, FPR_1

# The code that I got 1.0 / 1.0 for the original autograder system

# z_test = X_test[:, sensitive_attribute]

# TPR_0 = np.sum((y_pred[z_test == 0] == 1) & (y_test[z_test == 0] == 1)) / np.sum(y_test[z_test == 0] == 1 & (y_test[z_test == 0] == 1))
# FPR_0 = np.sum((y_pred[z_test == 0] == 0) & (y_test[z_test == 0] == 1)) / np.sum(y_test[z_test == 0] == 1)

# TPR_1 = np.sum((y_pred[z_test == 1] == 1) & (y_test[z_test == 1] == 1)) / np.sum(y_test[z_test == 1] == 1 & (y_test[z_test == 1] == 1))
# FPR_1 = np.sum((y_pred[z_test == 1] == 0) & (y_test[z_test == 1] == 1)) / np.sum(y_test[z_test == 1] == 1)

# TPR_0, TPR_1, FPR_0, FPR_1

"""5. Compute PPV_0, PPV_1, NPV_0, and NPV_1 (see "are fairness goals compatible" slides)."""

z_test = (X_test[:, sensitive_attribute] == 1)

tp_0 = np.sum((y_pred == 1) & (y_test == 1) & (z_test == 0))
fn_0 = np.sum((y_pred == 0) & (y_test == 1) & (z_test == 0))
fp_0 = np.sum((y_pred == 1) & (y_test == 0) & (z_test == 0))
tn_0 = np.sum((y_pred == 0) & (y_test == 0) & (z_test == 0))

tp_1 = np.sum((y_pred == 1) & (y_test == 1) & (z_test == 1))
fn_1 = np.sum((y_pred == 0) & (y_test == 1) & (z_test == 1))
fp_1 = np.sum((y_pred == 1) & (y_test == 0) & (z_test == 1))
tn_1 = np.sum((y_pred == 0) & (y_test == 0) & (z_test == 1))


PPV_0 = tp_0 / (tp_0 + fp_0)
PPV_1 = tp_1 / (tp_1 + fp_1)

NPV_0 = tn_0 / (tn_0 + fn_0)
NPV_1 = tn_1 / (tn_1 + fn_1)

PPV_0, PPV_1, NPV_0, NPV_1

"""6. Implement a "fairness through unawareness" classifier, i.e., don"t use Z in your feature vector. Find the classifier coefficient which undergoes the largest (absolute value) change compared to the classifier with the feature included, and report its new coefficient.

"""

X_train_new = np.delete(X_train, sensitive_attribute, axis=1)
X_test_new = np.delete(X_test, sensitive_attribute, axis=1)

new_model = LogisticRegression(C=1, max_iter=1000, random_state=42)
new_model.fit(X_train_new, y_train)

original_coeff = np.delete(model.coef_.flatten(), sensitive_attribute)
new_coeff = new_model.coef_.flatten()

coeff_changes = np.abs(original_coeff - new_coeff)
biggest_change_idx = np.argmax(coeff_changes)
new_coeff = new_coeff[biggest_change_idx]

new_coeff, biggest_change_idx

"""7. Report the demographic parity of the classifier after implementing the above intervention."""

y_pred_new = new_model.predict(X_test_new)

z_test = (X_test[:, sensitive_attribute] == 1)

positive_pred_z_0 = np.mean(y_pred_new[z_test == 0])
positive_pred_z_1 = np.mean(y_pred_new[z_test == 1])

new_parity = positive_pred_z_0 / positive_pred_z_1

new_parity

"""8. Report the Generalized False Positive Rate and Generalized False Negative Rate of your original (i.e., not the one with z excluded).

"""

# The code that I got 1.0 / 1.0 for the original autograder system

# y_pred = model.predict(X_test)

# GFPR = np.sum(y_pred[(y_test == 0)]) / np.sum(y_test == 0)
# GFNR = np.sum(1 - y_pred[(y_test == 1)]) / np.sum(y_test == 1)

# GFPR, GFNR

y_pred_proba = model.predict_proba(X_test)

GFPR = np.sum(y_pred_proba[(y_test == 0), 1]) / np.sum(y_test == 0)
GFNR = np.sum(1 - y_pred_proba[(y_test == 1), 1]) / np.sum(y_test == 1)

GFPR, GFNR

"""9. (harder, 2 marks) Changing the classifier threshold (much as you would to generate an ROC curve) will change the False Positive and False Negative rates for both groups (i.e., FP_0, FP_1, FN_0, FN_1). Implement a "fairness through unawareness" classifier like you did in Question 6 but instead use feature 19 (i.e., feat[18]) as the sensitive attribute. Using this classifier, find the (non-trivial) threshold that comes closest to achieving Treatment Equality, and report the corresponding values of FP_0, FP_1, FN_0, and FN_1."""

sensitive_attribute = 18
new_model = LogisticRegression(C=1, max_iter=1000, random_state=42)

X_train_new = np.delete(X_train, sensitive_attribute, axis=1)
X_test_new = np.delete(X_test, sensitive_attribute, axis=1)
sensitive_test = X_test[:, sensitive_attribute]

new_model.fit(X_train_new, y_train)

y_pred_proba = new_model.predict_proba(X_test_new)[:, 1]

z_test = (X_test[:, sensitive_attribute] == 1)

best_threshold = None
min_treatment_diff = float("inf")
FP_0, FP_1, FN_0, FN_1 = 0, 0, 0, 0

thresholds = np.linspace(0, 1, 1000)
for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)

    fp_0 = np.sum((y_pred == 1) & (y_test == 0) & (z_test == 0))
    fn_0 = np.sum((y_pred == 0) & (y_test == 1) & (z_test == 0))

    fp_1 = np.sum((y_pred == 1) & (y_test == 0) & (z_test == 1))
    fn_1 = np.sum((y_pred == 0) & (y_test == 1) & (z_test == 1))

    ratio_0 = fp_0 / fn_0 if fn_0 != 0 else float("inf")
    ratio_1 = fp_1 / fn_1 if fn_1 != 0 else float("inf")

    treatment_diff = abs(ratio_0 - ratio_1)

    if treatment_diff < min_treatment_diff:
        min_treatment_diff = treatment_diff
        best_threshold = threshold
        FP_0, FP_1 = fp_0, fp_1
        FN_0, FN_1 = fn_0, fn_1

FP_0, FP_1, FN_0, FN_1, best_threshold

answers = {
    "Q1": prevalence,           # prevalence
    "Q2": [prevalence_0, prevalence_1],  # prevalence_0, prevalence_1
    "Q3": parity,           # parity
    "Q4": [TPR_0, TPR_1, FPR_0, FPR_1], # TPR_0, TPR_1, FPR_0, FPR_1
    "Q5": [PPV_0, PPV_1, NPV_0, NPV_1], # PPV_0, PPV_1, NPV_0, NPV_1
    "Q6": [biggest_change_idx, new_coeff], # feature index, coefficient
    "Q7": new_parity,           # parity
    "Q8": [GFPR, GFNR],  # GFPR, GFNR
    "Q9": [FP_0, FP_1, FN_0, FN_1]  # FP_0, FP_1, FN_0, FN_1
}
answers