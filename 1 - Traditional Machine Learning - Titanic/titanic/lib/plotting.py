import numpy as np
from matplotlib import pyplot as plot
import seaborn as sns

from .preprocessing import impute_age


def plot_age_histogram(data_frame, title):
    plot.hist(x=data_frame, stacked=True, bins=80)
    plot.title(f'Age Distribution ({title})')
    plot.xlabel('Age (Years)')
    plot.ylabel('# of Passengers')


def plot_imputation_on_age(imputer, data_frame, imputer_kwargs=None):
    imputed_data_frame = impute_age(imputer, data_frame, imputer_kwargs)
    plot_age_histogram(
        imputed_data_frame.Age, f'imputed with {imputer.__name__}'
    )


def plot_title(data_frame, subject):
    data_frame['Title'].value_counts().plot(kind='bar')
    plot.title(f'Title Distribution ({subject})')
    plot.xlabel('Title')
    plot.ylabel('# of Passengers')


def plot_tsne(X_tsne, y):
    markers = 's', 'd'
    color_map = {0: 'red', 1: 'blue'}
    classes = {0: 'Died', 1: 'Survived'}
    for i, klass in enumerate(np.unique(y)):
        plot.scatter(
            x=X_tsne[y == klass, 0],
            y=X_tsne[y == klass, 1],
            c=color_map[i],
            marker=markers[i],
            label=classes[i]
        )
    plot.xlabel('X in t-SNE')
    plot.ylabel('Y in t-SNE')
    plot.legend(loc='upper left')
    plot.title('t-SNE visualization of training data')


def plot_classifier_comparison(classifier_comparison):
    sns.barplot(x='Dev Accuracy Mean', y='Name', data=classifier_comparison)
    plot.title('Machine Learning Algorithm Accuracy Score \n')
    plot.xlabel('Accuracy Score (%)')
    plot.ylabel('Algorithm')
