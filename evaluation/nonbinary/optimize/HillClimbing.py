import numpy as np

'''
In this implementation, the initial solution is randomly generated, and the function repeatedly generates a new random neighbor solution and decides whether to accept it based on its fitness.
The algorithm stops when no better solutions can be found by generating new neighbors.
It's important to note that the results of this algorithm may vary depending on the specific function you're trying to minimize, as well as the initial state of the algorithm.
Also, note that the algorithm may get stuck in local minima, in this case, it can be improved by using techniques such as simulated annealing which allows for "jumping out" of local minima with a certain probability, or by using a technique called random restart, where the algorithm is run multiple times with different random initial solutions to increase the chances of finding the global optimum.
'''

def f(x):
    # replace this with your own blackbox function
    return sum(x)

def hill_climbing(d):
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

'''
In this example, I've added a constraint to the Hill Climbing algorithm that the sum of x has to be equal to a user-given number n. This is done by checking the sum of the new solution after it's generated, if it doesn't meet the constraint, the algorithm will continue to generate a new solution.

It's important to note that the results of this algorithm may vary depending on the specific function you're trying to minimize and the specific parameter values you choose.
'''
def hill_climbing_constraint(d, n):
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

'''
In this example, the Hill Climbing algorithm is run multiple times with different random initial solutions, this is the random restarts technique. This increases the chances of finding the global optimum and also deals with the exploration-exploitation trade-off. The algorithm stops when no better solutions can be found. The best solution found among all restarts is returned.

It's important to note that the number of restarts and the initial temperature should be chosen based on the specific problem at hand, as well as the desired trade-offs between exploration and exploitation.
'''
    
def hill_climbing_constraint_random_restart(d, n, num_restarts):
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
