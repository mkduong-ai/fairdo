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


def penalized_discrimination(y, z, n_groups, agg_group='max', **kwargs):
    """
    Penalized discrimination function that combines the statistical parity and group missing penalty.
    
    Parameters
    ----------
    y: np.array
        The target variable.
    z: np.array
        The protected attribute.
    n_groups: int
        The number of groups.
    
    Returns
    -------
    float
        The penalized discrimination."""
    if agg_group=='sum':
        penalized_discrimination = statistical_parity_abs_diff_sum(y=y, z=z) + group_missing_penalty(z=z, n_groups=n_groups, agg_group=agg_group)
    elif agg_group=='max':
        penalized_discrimination = np.max([statistical_parity_abs_diff_max(y=y, z=z), group_missing_penalty(z=z, n_groups=n_groups, agg_group=agg_group)])
    else:
        raise ValueError("Invalid aggregation group. Supported values are 'sum' and 'max'.")
    return penalized_discrimination


def run_optimization(data, label, protected_attribute, n_groups,
                     pop_size, num_generations,
                     initializer, selection, crossover, mutation):
    # Initialize multi-objective optimization
    ga = partial(nsga2,
                 pop_size=pop_size,
                 num_generations=num_generations,
                 initialization=initializer,
                 selection=selection,
                 crossover=crossover,
                 mutation=mutation)
    
    # Initialize the wrapper class for custom preprocessors
    preprocessor_multi = MultiObjectiveWrapper(heuristic=ga,
                                               protected_attribute=protected_attribute,
                                               label=label,
                                               fitness_functions=[partial(penalized_discrimination, n_groups=n_groups), data_loss])
    
    # Fit and transform the data
    preprocessor_multi.fit_transform(dataset=data)

    return preprocessor_multi.get_pareto_front(return_baseline=True)


def save_pareto_plot(pf, baseline, filename):
    """
    Plot the Pareto front.

    Parameters
    ----------
    pf : ndarray, shape (n, 2)
        Pareto front data containing two objectives.
    baseline : ndarray, shape (1,-1)
        Baseline fitness values.
    """
    # Create a scatter plot of the Pareto front
    plt.figure(figsize=(4, 2.5))
    sns.scatterplot(x=pf[:, 0], y=pf[:, 1],
                    label='Pareto Front',
                    color='red')
    sns.scatterplot(x=baseline[:, 0], y=baseline[:, 1],
                    label='Baseline',
                    color='blue')
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Set axis labels
    plt.xlabel(r'Discrimination $\tilde{\psi}$')
    plt.ylabel(r'Data Loss $\mathcal{L}$')
    
    # Set plot title
    # plt.title('Pareto Front')

    # Save plot
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    
    # Show plot
    # plt.show()


def process_run(args):
    i, data, label, protected_attributes, n_groups, pop_size, num_generations, initializer, selection, crossover, mutation, ref_point, data_str = args
    pf, baseline = run_optimization(data, label, protected_attributes, n_groups,
                                    pop_size, num_generations,
                                    initializer, selection, crossover, mutation)
    ind = HV(ref_point=ref_point)
    hv = ind(pf)
    new_row = {'Trial': i,
               'Dataset': data_str,
               'Label': label,
               'Protected_Attributes': protected_attributes,
               'N_Groups': n_groups,
               'Initializer': initializer.__name__,
               'Selection': selection.__name__,
               'Crossover': crossover.__name__,
               'Mutation': mutation.__name__,
               'Hypervolume': hv,
               'Pareto_Front': pf,
               'Baseline': baseline}
    
    print(i)
    # Save Pareto plot
    if i == 0:
        if not os.path.exists(f'results/{data_str}'):
            os.makedirs(f'results/{data_str}')

        plot_filename = f'results/{data_str}/pareto_plot_{initializer.__name__}_{selection.__name__}_{crossover.__name__}_{mutation.__name__}.pdf'
        print(plot_filename)
        save_pareto_plot(pf, baseline, plot_filename)

    return new_row


def main():
    ref_point = np.array([1.0, 1.0])

    # number of runs
    n_runs = 10

    # settings
    pop_size = 100
    num_generations = 200

    initializers = [variable_initialization, random_initialization]
    selections = [elitist_selection_multi, tournament_selection_multi]
    crossovers = [uniform_crossover, onepoint_crossover]
    mutations = [bit_flip_mutation, shuffle_mutation]

    # Loading a sample database and encoding for appropriate usage
    # data is a pandas dataframe
    data_str = 'compas'
    data, label, protected_attributes = load_data(data_str, print_info=False)
    n_groups = len(data[protected_attributes[0]].unique())

    args_list = [(i, data, label, protected_attributes, n_groups, pop_size, num_generations, initializer, selection, crossover, mutation, ref_point, data_str)  
                 for initializer, selection, crossover, mutation in itertools.product(initializers, selections, crossovers, mutations)
                 for i in range(n_runs)]

    with ProcessPool() as pool:
        print('Number of processes:', pool.ncpus)
        results = pool.map(process_run, args_list)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{data_str}/optimization_results.csv', index=False)


def main_deprecated():
    ref_point = np.array([1.0, 1.0])

    # number of runs
    n_runs = 10

    # settings
    pop_size = 100
    num_generations = 200

    initializers = [variable_initialization, random_initialization]
    selections = [elitist_selection_multi, tournament_selection_multi]
    crossovers = [uniform_crossover, onepoint_crossover]
    mutations = [bit_flip_mutation, shuffle_mutation]

    # Loading a sample database and encoding for appropriate usage
    # data is a pandas dataframe
    data_str = 'compas'
    data, label, protected_attributes = load_data(data_str, print_info=False)
    n_groups = len(data[protected_attributes[0]].unique())

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['Trial', 'Dataset', 'Label', 'Protected_Attributes', 'N_Groups',
                                       'Initializer', 'Selection', 'Crossover', 'Mutation',
                                       'Hypervolume', 'Pareto_Front', 'Baseline'])
    
    for initializer, selection, crossover, mutation in itertools.product(initializers, selections, crossovers, mutations):
        for i in range(n_runs):
            pf, baseline = run_optimization(data, label, protected_attributes, n_groups,
                                pop_size, num_generations,
                                initializer, selection, crossover, mutation)

            ind = HV(ref_point=ref_point)
            hv = ind(pf)

            # Append results to DataFrame
            new_row = {'Trial': i,
            'Dataset': data_str,
            'Label': label,
            'Protected_Attributes': protected_attributes,
            'N_Groups': n_groups,
            'Initializer': initializer.__name__,
            'Selection': selection.__name__,
            'Crossover': crossover.__name__,
            'Mutation': mutation.__name__,
            'Hypervolume': hv,
            'Pareto_Front': pf,
            'Baseline': baseline}

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save Pareto plot
            if i == 0:
                if not os.path.exists(f'results/{data_str}'):
                    os.makedirs(f'results/{data_str}')

                plot_filename = f'results/{data_str}/pareto_plot_{initializer.__name__}_{selection.__name__}_{crossover.__name__}_{mutation.__name__}.pdf'
                print(plot_filename)
                save_pareto_plot(pf, baseline, plot_filename)

            print('Run:', i)

    # Save DataFrame to CSV
    results_df.to_csv(f'results/{data_str}/optimization_results.csv', index=False)


if __name__ == '__main__':
    main()
