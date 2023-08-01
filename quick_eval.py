# A helper function to wrap genetic algorithm methods and allow hyperparameter modifications

def genetic_algorithm_method_wrapper(crossover_type, population_size=100, generations=50):
    """
    Wraps the genetic algorithm method to allow hyperparameters modifications.

    Parameters
    ----------
    crossover_type: str
        Type of crossover in the genetic algorithm. 
        Should be 'one-point' or 'uniform'.
    population_size: int, optional
        The size of the population in the genetic algorithm.
    generations: int, optional
        The number of generations in the genetic algorithm.

    Returns
    -------
    method: function
        The genetic algorithm method with the specified hyperparameters.
    """

    def method(f, d):
        if crossover_type == 'one-point':
            return ga.genetic_algorithm_method(f, d, population_size, generations)
        elif crossover_type == 'uniform':
            return ga.genetic_algorithm_uniform_method(f, d, population_size, generations)
        else:
            raise ValueError("Invalid crossover type. Choose 'one-point' or 'uniform'.")
    
    return method

# Breaking down settings function

def create_save_path(data_str, objective_str):
    """
    Creates the save path for the experiment results.

    Parameters
    ----------
    data_str: str
        Name of the dataset.
    objective_str: str
        Objective function.

    Returns
    -------
    save_path: str
        The save path for the experiment results.
    """

    # create save path
    save_path = f'evaluation/results/nonbinary/{data_str}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = f'{save_path}/{data_str}_{objective_str}'
    
    return save_path

def setup_experiment(data_str, objective_str, n_runs):
    """
    Sets up the experiment.

    Parameters
    ----------
    data_str: str
        Name of the dataset.
    objective_str: str
        Objective function.
    n_runs: int
        Number of runs.

    Returns
    -------
    save_path: str
        The save path for the experiment results.
    disc_dict: dict
        Dictionary of discrimination measures.
    methods: dict
        Dictionary of methods.
    """

    # create objective functions
    disc_dict = {
        'Statistical Disparity Sum': nb_statistical_parity_sum_abs_difference,
        'Maximal Statistical Disparity': nb_statistical_parity_max_abs_difference,
        'NMI': nb_normalized_mutual_information,
        'Size': count_size,
        'Distinct Groups': count_groups,
        'Sanity Check': sanity_check}

    # create methods
    methods = {'Baseline (Original)': baseline.original_method,
               'Random Heuristic': baseline.random_method,
               'GA (1-Point Crossover)': genetic_algorithm_method_wrapper('one-point'),
               'GA (Uniform Crossover)': genetic_algorithm_method_wrapper('uniform')}

    # create save path
    save_path = create_save_path(data_str, objective_str)
    
    return save_path, disc_dict, methods

def run_and_save_experiment(data_str, objective_str, n_runs):
    """
    Runs the experiment and saves the results.

    Parameters
    ----------
    data_str: str
        Name of the dataset.
    objective_str: str
        Objective function.
    n_runs: int
        Number of runs.
    """

    # setup the experiment
    save_path, disc_dict, methods = setup_experiment(data_str, objective_str, n_runs)

    # run experiment
    results = run_experiments(data_str=data_str,
                              disc_dict=disc_dict,
                              methods=methods,
                              n_runs=n_runs,
                              objective_str=objective_str)

    # convert results to proper dataframe
    results_df = convert_results_to_dataframe(results)

    # save results
    results_df.to_csv(save_path + '.csv', index_label='index')

# Main function

def main():
    obj_strs = ['remove', 'add', 'remove_and_synthetic']
    data_strs = ['adult', 'compas']
    n_runs = 15
    for data_str in data_strs:
        print('------------------------------------')
        print(f'Running experiments for {data_str}...')
        for obj_str in obj_strs:
            print('------------------------------------')
            print(f'Running experiments for {obj_str}...')
            run_and_save_experiment(data_str=data_str, objective_str=obj_str, n_runs=n_runs)

if __name__ == "__main__":
    main()

