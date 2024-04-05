# standard library
from functools import partial
import itertools
import os
import time

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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, normalized_mutual_info_score

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
        disc_score = statistical_parity_abs_diff_sum(y=y,
                                                     z=z) + \
                                   group_missing_penalty(z=z,
                                                         n_groups=n_groups,
                                                         agg_group=agg_group)
    elif agg_group=='max':
        disc_score = np.max([statistical_parity_abs_diff_max(y=y,
                                                             z=z),
                                           group_missing_penalty(z=z,
                                                                 n_groups=n_groups,
                                                                 agg_group=agg_group,
                                                                 eps=eps)])#/(1+eps)
    else:
        raise ValueError("Invalid aggregation group. Supported values are 'sum' and 'max'.")
    return disc_score


# Single Objective
def weighted_loss(y, z, n_groups, dims, y_orig, z_orig, w=0.5, agg_group='max', eps=0.01, **kwargs):
    """
    A single objective function that combines the statistical parity and data loss.
    
    Parameters
    ----------
    y: np.array
        The target variable.
    z: np.array
        The protected attribute.
    dims: int
        The number of samples.
    
    Returns
    -------
    float
        The weighted fairness and quality of the data."""
    beta = penalized_discrimination(y=y_orig, z=z_orig, n_groups=n_groups, agg_group=agg_group, eps=eps)
    return w * penalized_discrimination(y=y, z=z, n_groups=n_groups, agg_group=agg_group, eps=eps)/beta +\
        (1-w) * data_loss(y=y, dims=dims)


def plot_results(results_df):
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x='Classifier', y='Accuracy', data=results_df, ax=ax[0])
    ax[0].set_title('Accuracy')
    sns.boxplot(x='Classifier', y='Statistical_Parity', data=results_df, ax=ax[1])
    ax[1].set_title('Statistical Parity')
    plt.show()


def preprocess_training_data_multi(data, label, protected_attributes, n_groups):
    # settings
    pop_size = 200
    num_generations = 400

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
                                               fitness_functions=[partial(penalized_discrimination,
                                                                          n_groups=n_groups),
                                                                  data_loss])
    
    # Parameters for beta normalization
    y_orig = data[label].to_numpy()
    z_orig = data[protected_attributes[0]].to_numpy()
    beta = penalized_discrimination(y=y_orig, z=z_orig, n_groups=n_groups)

    # Fit and transform the data, returns the data closest to the ideal solution
    preprocessor_multi.fit(dataset=data)
    data_multi = preprocessor_multi.transform(w=np.array([1/beta, 1]))

    # Return the fitness values of the returned data as well as the baseline
    best_fitness, baseline_fitness = preprocessor_multi.get_best_fitness(return_baseline=True)

    return data_multi, best_fitness, baseline_fitness


def preprocess_training_data_single(data, label, protected_attributes, n_groups):
    # settings
    pop_size = 200
    num_generations = 400
    
    ga = partial(genetic_algorithm,
             pop_size=pop_size,
             num_generations=num_generations,
             initialization=variable_initialization,
             crossover=onepoint_crossover,
             mutation=bit_flip_mutation)
    
    # Parameters for beta normalization
    y_orig = data[label].to_numpy()
    z_orig = data[protected_attributes[0]].to_numpy()

    # Initialize the wrapper class for custom preprocessors
    preprocessor = HeuristicWrapper(heuristic=ga,
                                protected_attribute=protected_attributes[0],
                                label=label,
                                fitness_functions=[partial(weighted_loss,
                                                           y_orig=y_orig,
                                                           z_orig=z_orig,
                                                           n_groups=n_groups)])
    
    # Fit and transform the data
    data_single = preprocessor.fit_transform(dataset=data)

    # Return the fitness values of the returned data as well as the baseline
    best_fitness, baseline_fitness = preprocessor.get_best_fitness(return_baseline=True)

    return data_single, best_fitness, baseline_fitness


def run_dataset_single_thread(data_str, approach='multi'):
    print(f'Running {data_str} with {approach} approach')
    # number of runs
    n_runs = 10

    # Loading a sample database and encoding for appropriate usage
    # data is a pandas dataframe
    data, label, protected_attributes = load_data(data_str, print_info=False)
    n_groups = len(data[protected_attributes[0]].unique())

    # Split the data before optimizing for fairness
    train_df, test_df = train_test_split(data, test_size=0.2,
                                         stratify=data[protected_attributes[0]],
                                         random_state=42)
    # print(train_df[protected_attributes[0]].value_counts())
    # print(len(train_df[protected_attributes[0]].unique()))

    results = []
    for i in range(n_runs):
        print(f'Run: {i}')
        # Optimize training data for fairness
        if approach == 'multi':
            start = time.time()
            fair_df, fitness, baseline_fitness = preprocess_training_data_multi(train_df, label, protected_attributes, n_groups)
            elapsed = time.time() - start
        else:
            start = time.time()
            fair_df, fitness, baseline_fitness = preprocess_training_data_single(train_df, label, protected_attributes, n_groups)
            elapsed = time.time() - start
        
        # Split data to features X and label y
        X_fair_train, y_fair_train = fair_df.loc[:, fair_df.columns!=label], fair_df[label]
        X_orig_train, y_orig_train = train_df.loc[:, train_df.columns!=label], train_df[label]
        X_test, y_test = test_df.loc[:, test_df.columns!=label], test_df[label]

        # Train and evaluate classifier
        classifiers = [SVC(), LogisticRegression(), RandomForestClassifier(), MLPClassifier()]

        for clf in classifiers:
            # Training data/Original data
            statistical_parity_train_fair = statistical_parity_abs_diff_max(y_fair_train, fair_df[protected_attributes[0]].to_numpy())
            statistical_parity_train = statistical_parity_abs_diff_max(train_df[label].to_numpy(), train_df[protected_attributes[0]].to_numpy())

            # Train and evaluate classifier on fair data
            clf.fit(X_fair_train, y_fair_train)
            # Metrics for classification
            y_pred = clf.predict(X_test)
            accuracy = clf.score(X_test, y_test)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            # Fairness metrics (No penalty because groups might be missing)
            statistical_parity = statistical_parity_abs_diff_max(y_pred, test_df[protected_attributes[0]].to_numpy())
            nmi = normalized_mutual_info_score(y_pred, test_df[protected_attributes[0]].to_numpy())

            # Train and evaluate classifier on original data
            clf.fit(X_orig_train, y_orig_train)
            # Metrics for classification
            y_pred = clf.predict(X_test)
            accuracy_orig = clf.score(X_test, y_test)
            balanced_accuracy_orig = balanced_accuracy_score(y_test, y_pred)
            f1_orig = f1_score(y_test, y_pred)
            roc_auc_orig = roc_auc_score(y_test, y_pred)
            # Fairness metrics (No penalty because groups might be missing)
            statistical_parity_orig = statistical_parity_abs_diff_max(y_pred, test_df[protected_attributes[0]].to_numpy())
            nmi_orig = normalized_mutual_info_score(y_pred, test_df[protected_attributes[0]].to_numpy())

            results.append({'Trial': i,
                            'Approach': approach,
                            'Time': elapsed,
                            'Dataset': data_str,
                            'Label': label,
                            'Protected_Attributes': protected_attributes,
                            'N_Groups': n_groups,
                            'Statistical_Parity_Train': statistical_parity_train,
                            'Statistical_Parity_Train_Fair': statistical_parity_train_fair,
                            'Len_Train': len(train_df),
                            'Len_Train_Fair': len(fair_df),
                            'Classifier': clf.__class__.__name__,
                            'Accuracy': accuracy,
                            'Balanced_Accuracy': balanced_accuracy,
                            'F1': f1,
                            'ROC_AUC': roc_auc,
                            'Statistical_Parity': statistical_parity,
                            'NMI': nmi,
                            'Accuracy_Orig': accuracy_orig,
                            'Balanced_Accuracy_Orig': balanced_accuracy_orig,
                            'F1_Orig': f1_orig,
                            'ROC_AUC_Orig': roc_auc_orig,
                            'Statistical_Parity_Orig': statistical_parity_orig,
                            'NMI_Orig': nmi_orig,
                            'Fitness': fitness,
                            'Baseline_Fitness': baseline_fitness})
        
            print(f'Classifier: {clf.__class__.__name__}')


    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{data_str}/{approach}_beta_classifier_results.csv', index=False)

    print(f'Saved results for {data_str} with {approach} approach to results/{data_str}/{approach}_classifier_results.csv')


def main():
    # Run for all datasets
    data_strs = ['adult', 'bank', 'compas']
    approaches = ['multi', 'single']

    with ProcessPool() as pool:
        print('Number of processes:', pool.ncpus)
        args_list = list(itertools.product(data_strs, approaches))

        print(args_list)
        pool.map(lambda args: run_dataset_single_thread(*args), args_list)


if __name__ == '__main__':
    main()
