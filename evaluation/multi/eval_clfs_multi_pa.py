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
from sklearn.preprocessing import LabelEncoder
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

# fairlearn
#from fairlearn.preprocessing import CorrelationRemover

# fairdo metrics
from fairdo.metrics import statistical_parity_abs_diff_max,\
    statistical_parity_abs_diff_sum,\
    data_loss, group_missing_penalty,\
    statistical_parity_abs_diff_multi,\
    statistical_parity_abs_diff_intersectionality


def dataset_intersectional_column(data, protected_attributes):
    """
    Parameters
    ----------
    data: pandas DataFrame

    Returns
    -------
    data: pandas DataFrame
        Returns a dataframe with an extra column of combines protected attributes
    """
    protected_attribute = 'pa_merged'
    
    # Initialize the protected attribute column with empty strings
    data[protected_attribute] = ''
    
    for col in protected_attributes:
        data[protected_attribute] += data[col].astype(str) + '_'
    
    return data, protected_attribute


def penalized_discrimination_multi(y, z, n_groups, agg_group='max', eps=0.01,
                                   intersectional=False, **kwargs):
    """
    Max SDP. We limit ourselves to 'max' discriminating attribute and 'max' disc. group.
    
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
    disc_score = np.max([statistical_parity_abs_diff_multi(y=y,
                                                        z=z,
                                                        agg_attribute=np.max,
                                                        agg_group=np.max),
                                    group_missing_penalty(z=z,
                                                            n_groups=n_groups,
                                                            agg_attribute='max',
                                                            agg_group='max',
                                                            eps=eps)])#/(1+eps)
    return disc_score


def weighted_loss_multi(y, z, n_groups, dims, y_orig, z_orig, w=0.5, agg_group='max', eps=0.01,
                        intersectional=False, **kwargs):
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
    #beta = penalized_discrimination_multi(y=y_orig, z=z_orig, n_groups=n_groups, agg_group=agg_group, eps=eps)
    #beta = 1/(1+eps)
    beta = 1
    return w * penalized_discrimination_multi(y=y, z=z, n_groups=n_groups, agg_group=agg_group, eps=eps,
                                              intersectional=intersectional)/beta +\
           (1-w) * data_loss(y=y, dims=dims)


def plot_results(results_df):
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x='Classifier', y='Accuracy', data=results_df, ax=ax[0])
    ax[0].set_title('Accuracy')
    sns.boxplot(x='Classifier', y='Statistical_Parity', data=results_df, ax=ax[1])
    ax[1].set_title('Statistical Parity')
    plt.show()


def preprocess_training_data_multi(data, label, protected_attributes, n_groups,
                                   intersectional):
    # settings
    pop_size = 20
    num_generations = 4
    eps = 0.01

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
                                               protected_attribute=protected_attributes,
                                               label=label,
                                               fitness_functions=[partial(penalized_discrimination_multi,
                                                                          n_groups=n_groups,
                                                                          eps=eps,
                                                                          intersectional=intersectional),
                                                                  data_loss])
    
    # Parameters for beta normalization
    # y_orig = data[label].to_numpy()
    # z_orig = data[protected_attributes[0]].to_numpy()
    # z_orig = data[protected_attributes].to_numpy()
    #beta = penalized_discrimination_multi(y=y_orig, z=z_orig, n_groups=n_groups,
    #                                      intersectional=intersectional)
    #beta = 1/(1+eps)
    beta = 1

    # Fit and transform the data, returns the data closest to the ideal solution
    preprocessor_multi.fit(dataset=data)
    # data_multi = preprocessor_multi.transform(w=np.array([1/beta, 1]))
    data_multi = preprocessor_multi.transform(w=np.array([beta, 1]))

    # Return the fitness values of the returned data as well as the baseline
    best_fitness, baseline_fitness = preprocessor_multi.get_best_fitness(return_baseline=True)

    return data_multi, best_fitness, baseline_fitness


def preprocess_training_data_single(data, label, protected_attributes, n_groups,
                                    intersectional):
    # settings
    pop_size = 20
    num_generations = 4
    
    ga = partial(genetic_algorithm,
             pop_size=pop_size,
             num_generations=num_generations,
             initialization=variable_initialization,
             crossover=onepoint_crossover,
             mutation=bit_flip_mutation,
             patience=num_generations)
    
    # Parameters for beta normalization
    y_orig = data[label].to_numpy()
    # z_orig = data[protected_attributes[0]].to_numpy()
    z_orig = data[protected_attributes].to_numpy()
    # Initialize the wrapper class for custom preprocessors
    preprocessor = HeuristicWrapper(heuristic=ga,
                                protected_attribute=protected_attributes,
                                label=label,
                                fitness_functions=[partial(weighted_loss_multi,
                                                           y_orig=y_orig,
                                                           z_orig=z_orig,
                                                           n_groups=n_groups,
                                                           intersectional=intersectional)])
    
    # Fit and transform the data
    data_single = preprocessor.fit_transform(dataset=data)

    # Return the fitness values of the returned data as well as the baseline
    best_fitness, baseline_fitness = preprocessor.get_best_fitness(return_baseline=True)

    return data_single, best_fitness, baseline_fitness


def run_dataset_single_thread(data_str, approach='multi'):
    print(f'Running {data_str} with {approach} approach')
    # number of runs
    n_runs = 2
    intersectional= True

    # Loading a sample database and encoding for appropriate usage
    # data is a pandas dataframe
    data, label, protected_attributes = load_data(data_str, multi_protected_attr=True, print_info=False)
    data, protected_attribute = dataset_intersectional_column(data, protected_attributes)
    #n_groups = data[protected_attributes].nunique().to_numpy()
    # Label encode the protected attribute
    le = LabelEncoder()
    data[protected_attribute] = le.fit_transform(data[protected_attribute])
    n_groups = data[protected_attribute].nunique()

    # Split the data before optimizing for fairness
    train_df, test_df = train_test_split(data, test_size=0.2,
                                         stratify=data[protected_attribute].to_numpy(),
                                         random_state=42)
    print(train_df[protected_attribute].value_counts())
    print(len(train_df[protected_attribute].unique()))

    results = []
    for i in range(n_runs):
        print(f'Run: {i}')
        # Optimize training data for fairness
        if approach == 'multi':
            start = time.perf_counter()
            fair_df, fitness, baseline_fitness = preprocess_training_data_multi(train_df, label, protected_attribute, n_groups,
                                                                                intersectional=intersectional)
            elapsed = time.perf_counter() - start
        elif approach == 'single':
            start = time.perf_counter()
            fair_df, fitness, baseline_fitness = preprocess_training_data_single(train_df, label, protected_attribute, n_groups,
                                                                                 intersectional=intersectional)
            elapsed = time.perf_counter() - start

        # Split data to features X and label y
        X_fair_train, y_fair_train = fair_df.loc[:, fair_df.columns!=label], fair_df[label]
        X_orig_train, y_orig_train = train_df.loc[:, train_df.columns!=label], train_df[label]
        X_test, y_test = test_df.loc[:, test_df.columns!=label], test_df[label]

        # Train and evaluate classifier
        # classifiers = [SVC(), LogisticRegression(), RandomForestClassifier(), MLPClassifier()]
        classifiers = [LogisticRegression()]

        for clf in classifiers:
            sdp = statistical_parity_abs_diff_multi
            # Training data/Original data
            statistical_parity_train_fair = sdp(y_fair_train, fair_df[protected_attribute].to_numpy())
            statistical_parity_train = sdp(y_orig_train, train_df[protected_attribute].to_numpy())

            # Train and evaluate classifier on fair data
            clf.fit(X_fair_train, y_fair_train)
            # Metrics for classification
            y_pred = clf.predict(X_test)
            accuracy = clf.score(X_test, y_test)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            # Fairness metrics (No penalty because groups might be missing)
            statistical_parity = sdp(y_pred, test_df[protected_attribute].to_numpy())
            nmi = normalized_mutual_info_score(y_pred, test_df[protected_attribute].to_numpy())

            # Train and evaluate classifier on original data
            clf.fit(X_orig_train, y_orig_train)
            # Metrics for classification
            y_pred = clf.predict(X_test)
            accuracy_orig = clf.score(X_test, y_test)
            balanced_accuracy_orig = balanced_accuracy_score(y_test, y_pred)
            f1_orig = f1_score(y_test, y_pred)
            roc_auc_orig = roc_auc_score(y_test, y_pred)
            # Fairness metrics (No penalty because groups might be missing)
            statistical_parity_orig = sdp(y_pred, test_df[protected_attribute].to_numpy())
            nmi_orig = normalized_mutual_info_score(y_pred, test_df[protected_attribute].to_numpy())

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
    # save results to csv
    if not os.path.exists(f'results/multi_pa/{data_str}'):
        os.makedirs(f'results/multi_pa/{data_str}')

    results_df.to_csv(f'results/multi_pa/{data_str}/{approach}_intersectional_classifier_results.csv', index=False)

    print(f'Saved results for {data_str} with {approach} approach to results/multi_pa/{data_str}/{approach}_intersectional_classifier_results.csv')


def main():
    # Run for all datasets
    data_strs = ['adult', 'bank', 'compas']
    data_strs = ['bank', 'compas']
    approaches = ['multi', 'single']

    with ProcessPool() as pool:
        print('Number of processes:', pool.ncpus)
        args_list = list(itertools.product(data_strs, approaches))

        print(args_list)
        pool.map(lambda args: run_dataset_single_thread(*args), args_list)
        #list(map(lambda args: run_dataset_single_thread(*args), args_list)) # single thread

if __name__ == '__main__':
    main()