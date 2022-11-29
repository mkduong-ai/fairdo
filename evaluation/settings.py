from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

dataset_pro_attributes_template = [('adult', 'sex'),
                                   ('compas', 'race'),
                                   ('german', 'foreign_worker'),
                                   ('german', 'sex'),
                                   ('bank', 'age')]
models_template = [LogisticRegression(),
                   DecisionTreeClassifier(),
                   RandomForestClassifier(),
                   SVC(probability=True),
                   MLPClassifier(),  # MLP does not support sample weight
                   KNeighborsClassifier()]  # KNN does not support sample weight
x_axes_template = ['Mutual Information',
                   'Normalized MI',
                   'Randomized Dependence Coefficient',
                   'Pearson Correlation',
                   'Statistical Parity Abs Diff',
                   'Disparate Impact', 'Disparate Impact Obj',
                   'Equal Opportunity Abs Diff',
                   'Predictive Equality Abs Diff',
                   'Average Odds Diff'
                   'Average Odds Error',
                   'Consistency',
                   'Consistency Obj']
y_axes_template = ['Accuracy', 'F1 Score', 'Balanced Accuracy', 'AUC']


def get_evaluation_config(config='comparison_preprocessors', frac=0.75, plot=False):
    models = [KNeighborsClassifier(),
              LogisticRegression(),
              DecisionTreeClassifier()]

    if frac < 1:
        filepath = f"results/MetricOptRemover"
    else:
        filepath = f"results/MetricOptGenerator"

    if config == 'comparison_preprocessors':
        dataset_pro_attributes = [('adult', 'sex'),
                                  ('bank', 'age'),
                                  ('compas', 'race'),
                                  ('german', 'foreign_worker'),
                                  ]

        if plot:
            models = list(map(lambda x: type(x).__name__, models))

        return dataset_pro_attributes, models, filepath
    elif config == 'fairness_agnostic':
        dataset_pro_attributes = [('compas', 'race')]

        metrics = [#'statistical_parity_absolute_difference',
                   'normalized_mutual_information',
                   'consistency_score_objective',
                   'disparate_impact_ratio_objective']

        if plot:
            models = list(map(lambda x: type(x).__name__, models))

        return dataset_pro_attributes, models, metrics, filepath
    elif config == 'quick':
        dataset_pro_attributes = [('compas', 'race')]

        if plot:
            models = list(map(lambda x: type(x).__name__, models))

        return dataset_pro_attributes, models, filepath
