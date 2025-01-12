############################################################################
# Modify code however you want, as long as you provide the functions below #
############################################################################

from sklearn import linear_model
import random
import numpy as np

random.seed(0)

#######################
# Part 1: Accuracy    #
#######################

# Baseline: Trivial model that includes a single feature, and the sensitive attribute


def p1model():
    return linear_model.LogisticRegression(
        C=2.0,
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
        # elif "AGE" in key:
        #     features += [1 if int(value) // 10 == i else 0 for i in range(8)]
        #     features.append(int(value) // 10)
        else:
            features.append(float(value))

    # features += [1 if int(z) == i else 0 for i in range(7)]
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


def p2data(data):
    newd = []
    one_count = 0
    others_count = 0
    for _, z, _ in data:
        one_count += 1 if z == 1 else 0
        others_count += 1 if z != 1 else 0

    min_value = min(one_count, others_count)
    select_others_count = 0
    for d, z, l in data:
        if z == 1:
            newd.append((d, z, l))
        elif z != 1 and select_others_count < min_value / 2.0:
            select_others_count += 1
            newd.append((d, z, l))
        else:
            continue
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
    model = p1model()
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


def p4labels(test_scores, dTest, zTest, threshold0=0.520, threshold1=0.480):
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
    dataTrain = p2data(data=dataTrain)
    X_train = [p1feat(d, z) for d, z, _ in dataTrain]
    y_train = [l for _, _, l in dataTrain]
    X_test = [p1feat(d, z) for d, z in zip(dTest, zTest)]
    model = p1model()
    model.fit(X_train, y_train)
    test_scores = [x[1] for x in model.predict_proba(X_test)]
    test_predictions = p4labels(test_scores, dTest, zTest, threshold0=0.470)
    return test_predictions
