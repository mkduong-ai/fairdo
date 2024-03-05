import numpy as np
import matplotlib.pyplot as plt
import time

from fairdo.optimize import nsga2

def function3(x):
    return np.sum(x) #* np.random.rand()

def function4(x):
    return -np.sum(x) #* np.random.rand()

# Define the number of dimensions
d = 10000

# Define the number of generations
num_generations = 100

# Define the population size
pop_size = 25

# Define the fitness functions to optimize
fitness_functions = [function3, function4]

# Run NSGA-II for each fitness function
solutions = []
#for f in fitness_functions:
#    solution, _ = nsga2([f], d, pop_size, num_generations)
#    solutions.append(solution)

start = time.time()
solutions, fitness, fronts = nsga2(fitness_functions, d, pop_size, num_generations, return_all_fronts=True)
print("Time:", time.time() - start)

# Plot the solutions
plt.figure(figsize=(5, 5))
# Plot all fronts
for i in range(len(fronts)):
    plt.scatter(fitness[fronts[i]][:, 0], fitness[fronts[i]][:, 1],
                label=f'Front {i+1}',
                s=15)
plt.xlabel('x')
plt.ylabel('Fitness')
plt.title('NSGA-II Optimization Results')
plt.legend()
plt.grid(True)
plt.show()
plt.close()