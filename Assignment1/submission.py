############################################################################
# Modify code however you want, as long as you provide the functions below #
############################################################################

from sklearn import linear_model
import random
import numpy as np
import subprocess
import sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
# from lightgbm import LGBMClassifier

random.seed(0)

#######################
# Part 1: Accuracy    #
#######################

# Baseline: Trivial model that includes a single feature, and the sensitive attribute


def p1model():
    # For LGBMClassifier
    # return LGBMClassifier(
    #     class_weight="balanced",
    #     n_estimators=200,
    #     learning_rate=0.05,
    #     max_depth=5,
    #     num_leaves=9,
    #     random_state=3407
    # )

    # For LogisticRegression
    return linear_model.LogisticRegression(
        C=5.0,
        max_iter=2000,
        class_weight="balanced"
    )


# d: a dictionary describing a training instance (excluding the sensitive attribute)
# z: sensitive attribute (True if married)
# return: a feature vector (list of floats)


def p1feat(d, z):
    features = []
    for key, value in d.items():
        if key == "ID":
            continue
        elif "LIMIT" in key or "AMT" in key:
            val = float(value)
            if val < 0:
                features.append(0)
            else:
                features.append(np.log10(val + 1))
        elif "SEX" in key:
            features.append(1 if int(value) == 2 else 0)
        elif "MARRIAGE" in key:
            features += [1 if int(value) == i else 0 for i in range(4)]
        elif "PAY" in key and "AMT" not in key:
            features += [1 if int(value) == i else 0 for i in range(-1, 10)]
        elif "AGE" in key:
            features += [1 if int(value) // 10 == i else 0 for i in range(2, 8)]
        else:
            features.append(float(value))

    features += [1 if int(z) == 1 else 0]
    return features

#########################################
# Part 2: Dataset-based intervention    #
#########################################

# Baseline: Just double all of the instances with a positive sensitive attribute


def p2model():
    return p1model()

# data: the dataset, which is a list of tuples of the form (d,z,l)
# d: feature dictionary (excluding sensitive attribute)
# z: sensitive attributes
# l: label
# return: a dataset in the same form


def p2data(data, hp=0):
    newd = []
    one_data = []
    others_data = []

    for d, z, l in data:
        if z == 1:
            one_data.append((d, z, l))
        else:
            others_data.append((d, z, l))

    one_count = len(one_data)
    others_count = len(others_data)

    if one_count < others_count:
        multiplier = others_count // one_count
        remainder = others_count % one_count + hp
        for _ in range(multiplier):
            newd.extend(one_data)
        newd.extend(one_data[:remainder])
    else:
        newd.extend(one_data)

    newd.extend(others_data)

    return newd

#########################################
# Problem 3: Model-based intervention   #
#########################################

# Baseline: Give instances with z=0 twice the sample weight in a logistic regressor


def p3feat(d):
    features = []
    for key, value in d.items():
        if key == "ID":
            continue
        elif "LIMIT" in key or "AMT" in key:
            val = float(value)
            if val < 0:
                features.append(0)
            else:
                features.append(np.log10(val + 1))
        elif "SEX" in key:
            features.append(1 if int(value) == 2 else 0)
        elif "MARRIAGE" in key:
            features += [1 if int(value) == i else 0 for i in range(4)]
        elif "PAY" in key and "AMT" not in key:
            features += [1 if int(value) == i else 0 for i in range(-1, 10)]
        elif "AGE" in key:
            features += [1 if int(value) // 10 == i else 0 for i in range(2, 8)]
        else:
            features.append(float(value))

    return features

# data: the dataset, which is a list of tuples of the form (d,z,l)
# d: feature dictionary (excluding sensitive attribute)
# z: sensitive attributes
# l: label
# return: a model (already fit, so that receiver can call mod.predict)


def p3model(data):
    X_train = [p3feat(d) for d, _, _ in data]
    y_train = [l for _, _, l in data]

    # For LogisticRegression
    model = linear_model.LogisticRegression(
        C=5.0,
        max_iter=1500,
        class_weight={0: 1.0, 1: 4.5}
    )

    # For LGBMClassifier
    # model = LGBMClassifier(
    #     class_weight={0: 1.0, 1: 4.5},
    #     n_estimators=200,
    #     learning_rate=0.05,
    #     max_depth=5,
    #     num_leaves=9,
    #     random_state=7
    # )

    # You can use any model you want, though it must have a "predict" function which takes a feature vector
    model.fit(X_train, y_train)

    return model


###########################################
# Problem 4: Post-processing intervention #
###########################################

# Baseline: Perturb per-group thresholds by a bit

# test_scores: scores (probability estimates) for your classifier from Part 1
# dTest: the test data (list of dictionaries, i.e., just the features)
# zTest: list of sensitive attributes (list of bool)
# return: list of predictions (list of bool)


def p4labels(test_scores, dTest, zTest, threshold0=0.510, threshold1=0.480):
    predictions = []
    for s, z in zip(test_scores, zTest):
        if not z:
            if s > threshold0:
                predictions.append(1)
            else:
                predictions.append(0)
        if z:
            if s > threshold1:
                predictions.append(1)
            else:
                predictions.append(0)

    return predictions

########################################################
# Problem 5: Optimize p-rule, subject to accuracy > X% #
########################################################

# Baseline: Reuse solution from Part 1 (i.e., no improvement in model fairness)

# dataTrain: the dataset, which is a tuple of (ds,z,l) (as in other functions)
# dTest: the test data (list of dictionaries, i.e., just the features)
# zTest: the test sensitive attributes (list of bool)


def p5(dataTrain, dTest, zTest):
    dataTrain = p2data(data=dataTrain, hp=-500)

    X_train = [p1feat(d, z) for d, z, _ in dataTrain]
    y_train = [l for _, _, l in dataTrain]
    X_test = [p1feat(d, z) for d, z in zip(dTest, zTest)]

    # For LogisticRegression
    model = p1model()

    # For LGBMClassifier
    # model = LGBMClassifier(
    #     class_weight="balanced",
    #     n_estimators=200,
    #     learning_rate=0.05,
    #     max_depth=5,
    #     num_leaves=9,
    #     random_state=7
    # )

    model.fit(X_train, y_train)

    test_scores = [x[1] for x in model.predict_proba(X_test)]

    # For LogisticRegression
    test_predictions = p4labels(test_scores, dTest, zTest, threshold0=0.495)

    # For LGBMClassifier
    # test_predictions = p4labels(test_scores, dTest, zTest, threshold0=0.510, threshold1=0.490)

    return test_predictions
