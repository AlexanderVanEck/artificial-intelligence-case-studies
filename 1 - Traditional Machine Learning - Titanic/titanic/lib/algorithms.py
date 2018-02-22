import pandas as pd
from sklearn import (
    discriminant_analysis, ensemble, gaussian_process, linear_model,
    naive_bayes, neighbors, svm, tree, model_selection)
from xgboost import XGBClassifier

classifiers = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(max_iter=5),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(max_iter=5),
    linear_model.Perceptron(max_iter=5),

    # Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    # xgboost
    XGBClassifier()
]


def compare_classifiers(X, y):
    # I initialise a shuffle split class to split our dataset 10 times
    # and each batch will include 90% of the dataset.
    # This makes the classifiers more robust since we'll 'rotate' the
    # training data multiple times.
    shuffle_split_class = model_selection.ShuffleSplit(
        n_splits=10,
        test_size=0.3,
        train_size=0.6,
        random_state=0
    )

    classifier_comparison = []
    for classifier in classifiers:
        cross_validation = model_selection.cross_validate(
            classifier, X, y, cv=shuffle_split_class, return_train_score=True
        )

        classifier_output = {
            'Name': classifier.__class__.__name__,
            'Train Accuracy Mean': cross_validation['train_score'].mean(),
            'Dev Accuracy Mean': cross_validation['test_score'].mean(),
            'Dev Accuracy 3*STD': cross_validation['test_score'].std() * 3,
            'Time': cross_validation['fit_time'].mean(),
        }

        classifier_comparison.append(classifier_output)

    classifier_comparison = pd.DataFrame(
        classifier_comparison, columns=classifier_comparison[0].keys()
    )
    classifier_comparison.sort_values(
        by=['Dev Accuracy Mean'], ascending=False, inplace=True
    )
    return classifier_comparison
