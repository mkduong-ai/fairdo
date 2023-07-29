import numpy as np

from fado.utils.penalty import relative_difference_penalty


def acceptance_probability(delta, temperature):
    return np.exp(-delta / temperature)


def simulated_annealing_constraint(f, d, n, T_max, T_min, cooling_rate, max_iter=1000,
                                   penalty=relative_difference_penalty):
    """

    Parameters
    ----------
    f: callable
        The function to be minimized
    d: int
        dimension of the binary vector
    n: int
        constraint on the number of 1s in the binary vector
    T_max: float
        The initial temperature, which should be set high enough to allow the algorithm to
        escape local optima and explore the search space.
    T_min: float
        The final temperature, which should be set low enough to ensure that the algorithm has
        converged to a local minimum.
    cooling_rate: float
        The temperature decay rate, which should be set between 0 and 1. A value of 0.9 is often used.
    max_iter: int
        The maximum number of iterations to perform. This is used to prevent the algorithm from running
        indefinitely if it fails to converge.
    penalty: callable
        The penalty function that penalizes the fitness of a solution if it does not satisfy the constraint
        Parameters: solution (np.array), n (int)
        Returns: float

    Returns
    -------
    current_solution: np.array
        The best solution found by the algorithm
    current_fitness: float
        The fitness of the best solution found by the algorithm
    """

    # Initialize the current solution randomly
    current_solution = np.random.randint(2, size=d)
    current_fitness = f(current_solution)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    temperature = T_max
    iteration = 0
    # Repeat until the temperature reach the minimum
    while temperature > T_min and iteration < max_iter:
        # Generate a random neighbor
        new_solution = current_solution.copy()
        i = np.random.randint(d)
        new_solution[i] = 1 - new_solution[i]
        new_fitness = f(new_solution)
        # check the constraint
        if n != 0 and sum(new_solution) != n:
            new_fitness += penalty(new_solution, n)
        # Accept the new solution with a probability
        delta = new_fitness - current_fitness # if delta < 0, new solution is better
        if delta < 0 or acceptance_probability(delta, temperature) > np.random.rand():
            # Acceptance probability already includes the delta < 0. Only checked for clarity.
            current_solution = new_solution
            current_fitness = new_fitness
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        temperature *= cooling_rate  # decrease the temperature
        iteration += 1
    return best_solution, best_fitness


def simulated_annealing(f, d, T_max, T_min, cooling_rate, max_iter=1000):
    """

    Parameters
    ----------
    f: callable
        The function to be minimized
    d: int
        dimension of the binary vector
    T_max: float
        The initial temperature, which should be set high enough to allow the algorithm to
        escape local optima and explore the search space.
    T_min: float
        The final temperature, which should be set low enough to ensure that the algorithm has
        converged to a local minimum.
    cooling_rate: float
        The temperature decay rate, which should be set between 0 and 1. A value of 0.9 is often used.
    max_iter: int
        The maximum number of iterations to perform. This is used to prevent the algorithm from running
        indefinitely if it fails to converge.

    Returns
    -------
    current_solution: np.array
        The best solution found by the algorithm
    current_fitness: float
        The fitness of the best solution found by the algorithm
    """

    # Initialize the current solution randomly
    return simulated_annealing_constraint(f=f, d=d, n=0,
                                          T_max=T_max, T_min=T_min,
                                          cooling_rate=cooling_rate,
                                          max_iter=max_iter)


def simulated_annealing_method(f, d):
    """
    Parameters
    ----------
    f: callable
        The function to be minimized
    d: int
        dimension of the binary vector
    Returns
    -------
    current_solution: np.array
        The best solution found by the algorithm
    current_fitness: float
        The fitness of the best solution found by the algorithm
    """
    return simulated_annealing(f, d=d, T_max=1, T_min=1e-6, cooling_rate=0.99, max_iter=5000)
