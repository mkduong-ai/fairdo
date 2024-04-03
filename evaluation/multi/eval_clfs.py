# standard library
from functools import partial
import itertools
import os

# third party
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pathos.multiprocessing import ProcessPool
# plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# import ML metrics
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# fairdo package
from fairdo.utils.dataset import load_data
# everything needed for custom preprocessing
from fairdo.preprocessing import MultiObjectiveWrapper, HeuristicWrapper
from fairdo.optimize.multi import nsga2
from fairdo.optimize.single import genetic_algorithm
from fairdo.optimize.geneticoperators import variable_initialization, random_initialization,\
    elitist_selection, elitist_selection_multi, tournament_selection_multi,\
    uniform_crossover, onepoint_crossover, no_crossover, \
    fractional_flip_mutation, shuffle_mutation,\
    bit_flip_mutation
# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max,\
    statistical_parity_abs_diff_sum,\
    data_loss, group_missing_penalty


# Penalized discrimination
def penalized_discrimination(y, z, n_groups, agg_group='max', eps=0.01,**kwargs):
    """
    Penalized discrimination function that combines the statistical parity and group missing penalty.
    
    Parameters
    ----------
    y: np.array
        The target variable.
    z: np.array
        The protected attribute.
    
    Returns
    -------
    float
        The penalized discrimination."""
    if agg_group=='sum':
        penalized_discrimination = statistical_parity_abs_diff_sum(y=y,
                                                                   z=z) + \
                                   group_missing_penalty(z=z,
                                                         n_groups=n_groups,
                                                         agg_group=agg_group)
    elif agg_group=='max':
        penalized_discrimination = np.max([statistical_parity_abs_diff_max(y=y,
                                                                           z=z),
                                           group_missing_penalty(z=z,
                                                                 n_groups=n_groups,
                                                                 agg_group=agg_group,
                                                                 eps=eps)])/(1+eps)
    else:
        raise ValueError("Invalid aggregation group. Supported values are 'sum' and 'max'.")
    return penalized_discrimination


def plot_results(results_df):
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x='Classifier', y='Accuracy', data=results_df, ax=ax[0])
    ax[0].set_title('Accuracy')
    sns.boxplot(x='Classifier', y='Statistical_Parity', data=results_df, ax=ax[1])
    ax[1].set_title('Statistical Parity')
    plt.show()


def preprocess_training_data(data, label, protected_attributes, n_groups):
    # settings
    pop_size = 100
    num_generations = 200

    # Setting up pre-processor (Best settings from previous experiment)
    ga = partial(nsga2,
                pop_size=pop_size,
                num_generations=num_generations,
                initialization=variable_initialization,
                selection=elitist_selection_multi,
                crossover=onepoint_crossover,
                mutation=bit_flip_mutation)

    # Optimize training data for fairness
    # Initialize the wrapper class for custom preprocessors
    preprocessor_multi = MultiObjectiveWrapper(heuristic=ga,
                                               protected_attribute=protected_attributes[0],
                                               label=label,
                                               fitness_functions=[partial(penalized_discrimination, n_groups=n_groups),
                                                                  data_loss])
    
    # Fit and transform the data, returns the data closest to the ideal solution
    data_multi = preprocessor_multi.fit_transform(dataset=data)

    # Return the fitness values of the returned data as well as the baseline
    index_best = preprocessor_multi.best_index
    pf, baseline_fitness = preprocessor_multi.get_pareto_front(return_baseline=True)

    return data_multi, pf[index_best], baseline_fitness


def main():
    # number of runs
    n_runs = 10

    # Loading a sample database and encoding for appropriate usage
    # data is a pandas dataframe
    data_str = 'compas'
    data, label, protected_attributes = load_data(data_str, print_info=False)
    n_groups = len(data[protected_attributes[0]].unique())

    # Split the data before optimizing for fairness
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    for i in range(n_runs):
        # Optimize training data for fairness
        fair_df, fitness, baseline_fitness = preprocess_training_data(train_df, label, protected_attributes, n_groups)

        # Train and evaluate classifier
        classifiers = [SVC(), LogisticRegression(), RandomForestClassifier(), MLPClassifier()]

        results = []
        for clf in classifiers:
            # Split data to features X and label y
            X_fair_train, y_fair_train = fair_df.loc[:, fair_df.columns!=label], fair_df[label]
            X_orig_train, y_orig_train = train_df.loc[:, train_df.columns!=label], train_df[label]
            X_test, y_test = test_df.loc[:, test_df.columns!=label], test_df[label]

            # Train and evaluate classifier on fair data
            clf.fit(X_fair_train, y_fair_train)
            accuracy = clf.score(X_test, y_test)
            balanced_accuracy = balanced_accuracy_score(y_test, clf.predict(X_test))
            f1 = f1_score(y_test, clf.predict(X_test))
            roc_auc = roc_auc_score(y_test, clf.predict(X_test))

            # Train and evaluate classifier on original data
            clf.fit(X_orig_train, y_orig_train)
            accuracy_orig = clf.score(X_test, y_test)
            balanced_accuracy_orig = balanced_accuracy_score(y_test, clf.predict(X_test))
            f1_orig = f1_score(y_test, clf.predict(X_test))
            roc_auc_orig = roc_auc_score(y_test, clf.predict(X_test))

            results.append({'Trial': i,
                            'Dataset': data_str,
                            'Label': label,
                            'Protected_Attributes': protected_attributes,
                            'N_Groups': n_groups,
                            'Classifier': clf.__class__.__name__,
                            'Accuracy': accuracy,
                            'Balanced_Accuracy': balanced_accuracy,
                            'F1': f1,
                            'ROC_AUC': roc_auc,
                            'Accuracy_Orig': accuracy_orig,
                            'Balanced_Accuracy_Orig': balanced_accuracy_orig,
                            'F1_Orig': f1_orig,
                            'ROC_AUC_Orig': roc_auc_orig,
                            'Fitness': fitness,
                            'Baseline_Fitness': baseline_fitness})


    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{data_str}/classifier_results.csv', index=False)


if __name__ == '__main__':
    main()
