from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def comparison_preprocessors(model_string_bool=False):
    dataset_pro_attributes = [('adult', 'sex'),
                              ('compas', 'race'),
                              # ('german', 'foreign_worker'),
                              ('bank', 'age')]

    models = [KNeighborsClassifier(),
              LogisticRegression(),
              DecisionTreeClassifier()]

    if model_string_bool:
        models = list(map(lambda x: type(x).__name__, models))

    return dataset_pro_attributes, models


def fairness_agnostic(model_string_bool=False):
    dataset_pro_attributes = [('adult', 'sex'),
                              ('compas', 'race'),
                              # ('german', 'foreign_worker'),
                              ('bank', 'age')]

    models = [KNeighborsClassifier(),
              LogisticRegression(),
              DecisionTreeClassifier()]

    if model_string_bool:
        models = list(map(lambda x: type(x).__name__, models))

    return models
