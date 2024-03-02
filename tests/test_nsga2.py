import numpy as np
import matplotlib.pyplot as plt

from fairdo.optimize import nsga2

def function1(x):
    return x**2

def function2(x):
    return (x-2)**2

# Define the range of x values
x_min = -5
x_max = 5

# Define the number of dimensions
d = 1

# Define the number of generations
num_generations = 100

# Define the population size
pop_size = 10

# Define the fitness functions to optimize
fitness_functions = [function1, function2]

# Run NSGA-II for each fitness function
solutions = []
#for f in fitness_functions:
#    solution, _ = nsga2([f], d, pop_size, num_generations)
#    solutions.append(solution)

solutions, _ = nsga2([function1, function2], d, pop_size, num_generations)


# Plot the solutions
plt.figure(figsize=(10, 5))
for i, solution in enumerate(solutions):
    plt.scatter(solution, fitness_functions[i](solution), label=f'Function {i+1}')
plt.xlabel('x')
plt.ylabel('Fitness')
plt.title('NSGA-II Optimization Results')
plt.legend()
plt.grid(True)
plt.show()
