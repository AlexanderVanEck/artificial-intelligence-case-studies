import pandas as pd
from fancyimpute import KNN
from sklearn.preprocessing import LabelEncoder


class ImputationCollector:
    """
    This imputation collector functions as a simple list of callables that
    can be added to and later transform a data_frame.

    It comes in handy when you have multiple dataframes that need to be
    imputed by the same operations.
    """
    operations = []

    def add_operation(self, operation):
        """
        Add an operation to the imputation operations list.

        :param operation: a callable
        :return:
        """
        self.operations.append(operation)

    def transform(self, data_frame):
        """
        Transform a pd.DataFrame through every callable in the imputation
        operations list.

        :param data_frame: a data_frame to transform
        :return: a transformed data_frame
        :rtype: pd.DataFrame
        """
        for operation in self.operations:
            data_frame = operation(data_frame)

        return data_frame


def impute_embarked(data_frame):
    data_frame.Embarked.fillna('S', inplace=True)
    return data_frame


def impute_fare(data_frame):
    data_frame.Fare.fillna(data_frame[(data_frame.Embarked == 'S') & (
            data_frame.Pclass == 3)].Fare.mean(), inplace=True)
    return data_frame


def impute_age(imputer, data_frame, imputer_kwargs=None):
    # Copy the data_frame so we don't change the original
    data_frame = data_frame.copy()

    label_encoder = LabelEncoder()

    # the imputation library only allows numeric values, so we convert our
    # categorical values to numbers here.
    data_frame['Sex'] = label_encoder.fit_transform(data_frame['Sex'])
    data_frame['Embarked'] = label_encoder.fit_transform(
        data_frame['Embarked'])

    related_columns = ['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare',
                       'Embarked']

    if not imputer_kwargs:
        imputer_kwargs = {}

    return pd.DataFrame(
        imputer(verbose=False, **imputer_kwargs).complete(
            data_frame[related_columns]
        ),
        columns=related_columns
    )


def impute_age_with_knn(data_frame):
    imputed_data_frame = impute_age(KNN, data_frame, imputer_kwargs={'k': 5})
    data_frame.Age = imputed_data_frame.Age
    return data_frame


def impute_cabin(data_frame):
    data_frame.Cabin.fillna('U', inplace=True)
    return data_frame


def prepare_dummies_embarked_and_gender(data_frame):
    data_frame = pd.get_dummies(
        data_frame, columns=['Sex', 'Embarked'], drop_first=True)
    return data_frame


def prepare_age_and_fare_buckets(data_frame):
    data_frame.Fare = pd.qcut(data_frame['Fare'], 4)
    data_frame.Age = pd.cut(data_frame['Age'].astype(int), 5)

    label_encoder = LabelEncoder()
    data_frame['Fare'] = label_encoder.fit_transform(data_frame['Fare'])
    data_frame['Age'] = label_encoder.fit_transform(data_frame['Age'])
    return data_frame


def prepare_features_on_importance(data_frame):
    replace_features = {
        'Title': {
            'Rare': 4,
            'Rank': 3,
            'Married': 2,
            'Single': 1,
            'None': 0,
        },
        'Port': {
            'P': 2,
            'S': 1,
            'U': 0,
        },
        'Deck': {
            'G': 8,
            'F': 7,
            'E': 6,
            'D': 5,
            'C': 4,
            'B': 3,
            'A': 2,
            'T': 1,
            'U': 0,
        },

    }
    data_frame.replace(replace_features, inplace=True)
    return data_frame
