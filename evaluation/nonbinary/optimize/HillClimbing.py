import numpy as np


def f(x):
    # replace this with your own blackbox function
    return sum(x)


def hill_climbing(f, d):
    """
    An initial solution is randomly generated, and the function repeatedly generates a new random
    neighbor solution and decides whether to accept it based on its fitness.
    The algorithm stops when no better solutions can be found by generating new neighbors.

    Parameters
    ----------
    f: callable
        The function to be minimized
    d: int
        dimension of the binary vector
    Returns
    -------
    current_solution: np.array
    """
    # Initialize the current solution randomly
    current_solution = np.random.randint(2, size=d)
    current_fitness = f(current_solution)
    # Repeat until no better solutions can be found
    while True:
        improved = False
        for i in range(d):
            # Generate a random neighbor
            new_solution = current_solution.copy()
            new_solution[i] = 1 - new_solution[i]
            new_fitness = f(new_solution)
            # Check if the new solution is better
            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness
                improved = True
                break
        if not improved:
            break
    return current_solution, current_fitness


def hill_climbing_constraint(f, d, n=0):
    """
    An initial solution is randomly generated, and the function repeatedly generates a new random
    neighbor solution and decides whether to accept it based on its fitness.
    The algorithm stops when no better solutions can be found by generating new neighbors.
    Here a constraint is added to the problem, that the number of 1s in the binary vector should be equal to n.
    If a neighbor solution does not satisfy the constraint, it is discarded.

    Parameters
    ----------
    f: callable
        The function to be minimized
    d:  int
        dimension of the binary vector
    n: int
        constraint on the number of 1s in the binary vector

    Returns
    -------
    current_solution: np.array
    """
    # Initialize the current solution randomly
    current_solution = np.random.randint(2, size=d)
    current_fitness = f(current_solution)
    # Repeat until no better solutions can be found
    while True:
        improved = False
        for i in range(d):
            # Generate a random neighbor
            new_solution = current_solution.copy()
            new_solution[i] = 1 - new_solution[i]
            if sum(new_solution) != n:
                continue
            new_fitness = f(new_solution)
            # Check if the new solution is better
            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness
                improved = True
                break
        if not improved:
            break
    return current_solution, current_fitness


def hill_climbing_constraint_random_restart(f, d, n, num_restarts):
    """
    An initial solution is randomly generated, and the function repeatedly generates a new random
    neighbor solution and decides whether to accept it based on its fitness.
    The algorithm stops when no better solutions can be found by generating new neighbors.
    Here a constraint is added to the problem, that the number of 1s in the binary vector should be equal to n.
    If a neighbor solution does not satisfy the constraint, it is discarded.
    The algorithm is restarted with a new random initial solution num_restarts times to escape local minima.

    Parameters
    ----------
    f: callable
        The function to be minimized
    d:  int
        dimension of the binary vector
    n: int
        constraint on the number of 1s in the binary vector
    num_restarts: int
        number of times the algorithm is restarted with a new random initial solution

    Returns
    -------
    best_solution: np.array
    """
    best_solution = None
    best_fitness = float('inf')
    for i in range(num_restarts):
        # Initialize the current solution randomly
        current_solution = np.random.randint(2, size=d)
        current_fitness = f(current_solution)
        # Repeat until no better solutions can be found
        while True:
            improved = False
            for i in range(d):
                # Generate a random neighbor
                new_solution = current_solution.copy()
                new_solution[i] = 1 - new_solution[i]
                if sum(new_solution) != n:
                    continue
                new_fitness = f(new_solution)
                # Check if the new solution is better
                if new_fitness < current_fitness:
                    current_solution = new_solution
                    current_fitness = new_fitness
                    improved = True
                    break
            if not improved:
                break
        if current_fitness < best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness
    return best_solution, best_fitness
