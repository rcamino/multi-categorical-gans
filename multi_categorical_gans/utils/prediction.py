import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score, f1_score


def prediction_score(train_X, train_y, test_X, test_y, metric, model):
    # if the train labels are always the same
    values_train = set(train_y)
    if len(values_train) == 1:
        # predict always that value
        only_value_train = list(values_train)[0]
        test_pred = np.ones_like(test_y) * only_value_train

    # if the train labels have different values
    else:
        # create the model
        if model == "random_forest_classifier":
            m = RandomForestClassifier(n_estimators=10)
        elif model == "logistic_regression":
            m = LogisticRegression()
        else:
            raise Exception("Invalid model name.")

        # fit and predict
        m.fit(train_X, train_y)
        test_pred = m.predict(test_X)

    # calculate the score
    if metric == "f1":
        return f1_score(test_y, test_pred)
    elif metric == "accuracy":
        return accuracy_score(test_y, test_pred)
    else:
        raise Exception("Invalid metric name.")
