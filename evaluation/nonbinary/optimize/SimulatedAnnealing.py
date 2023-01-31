import random
import numpy as np
import math

'''
The function takes in three parameters:

d: the dimension of the binary vector
T_max: the initial temperature
T_min: the final temperature
alpha : the temperature decay rate
In this implementation, the initial solution is randomly generated, and the function repeatedly generates a new random
neighbor solution and decides whether to accept it based on its fitness and the current temperature.
The temperature is decreased after each iteration using the decay rate.
The algorithm stops when the temperature reaches the minimum.

It's important to note that the results of this algorithm may vary depending on the specific parameter
values you choose (e.g. T_max, T_min, alpha), as well as the specific function you're trying to minimize.

Also, note that the stopping criterion can be improved. For example, it could be based on the number of iterations,
or based on the improvement of the solution.
'''


def f(x):
    # replace this with your own blackbox function
    return sum(x)


def simulated_annealing(f, d, T_max, T_min, alpha, max_iter=1000):
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
    alpha: float
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
    current_solution = np.random.randint(2, size=d)
    current_fitness = f(current_solution)
    T = T_max
    iter = 0
    # Repeat until the temperature reach the minimum
    while T > T_min and iter < max_iter:
        for i in range(d):
            # Generate a random neighbor
            new_solution = current_solution.copy()
            new_solution[i] = 1 - new_solution[i]
            new_fitness = f(new_solution)
            # Accept the new solution with a probability
            delta = new_fitness - current_fitness
            if delta < 0 or math.exp(-delta/T) > random.random():
                current_solution = new_solution
                current_fitness = new_fitness
        T = T*alpha  # decrease the temperature
    return current_solution, current_fitness


def simulated_annealing_method(f, dims):
    """
    Parameters
    ----------
    f: callable
        The function to be minimized
    dims: int
        dimension of the binary vector
    Returns
    -------
    current_solution: np.array
        The best solution found by the algorithm
    current_fitness: float
        The fitness of the best solution found by the algorithm
    """
    return simulated_annealing(f, d=dims, T_max=1, T_min=1e-6, alpha=0.95, max_iter=1000)


def simulated_annealing_constraint(d, num_steps, n):
    # Initialize the current solution randomly
    current_solution = np.random.randint(2, size=d)
    current_fitness = f(current_solution)
    # Set the initial temperature
    temperature = 1
    # Set the cooling rate
    cooling_rate = 0.99
    # Repeat until the temperature is low
    for step in range(num_steps):
        # Generate a random neighbor
        new_solution = current_solution.copy()
        i = np.random.randint(d)
        new_solution[i] = 1 - new_solution[i]
        new_fitness = f(new_solution)
        # check the constraint
        if sum(new_solution) != n:
            new_fitness += penalty(new_solution, n)
        # Check if the new solution is better
        delta = new_fitness - current_fitness
        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
            current_solution = new_solution
            current_fitness = new_fitness
        temperature *= cooling_rate
    return current_solution, current_fitness


def penalty(x, n):
    # Not normalized
    return abs(sum(x) - n)
