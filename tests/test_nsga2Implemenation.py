import numpy as np

# Define NSGA-II class
class NSGA2:
    def __init__(self, population_size, num_generations, num_objectives, num_variables, crossover_rate=0.9, mutation_rate=0.1):
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_objectives = num_objectives
        self.num_variables = num_variables
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        return np.random.rand(self.population_size, self.num_variables)

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.num_variables - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        mutation_point = np.random.randint(0, self.num_variables)
        individual[mutation_point] = np.random.rand()
        return individual

    def fast_nondominated_sort(self, population, objectives):
        fronts = [[]]
        S = [[] for _ in range(len(population))]
        n = [0 for _ in range(len(population))]
        rank = [0 for _ in range(len(population))]
        front_count = {}

        for p, obj in enumerate(objectives):
            for q, obj2 in enumerate(objectives):
                if all(obj <= obj2) and any(obj < obj2):
                    S[p].append(q)
                elif all(obj2 <= obj) and any(obj2 < obj):
                    n[p] += 1

            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        for i, front in enumerate(fronts):
            front_count[i] = len(front)

        return fronts, rank, front_count

    def crowding_distance_assignment(self, front, objectives):
        num_individuals = len(front)
        crowding_distance = np.zeros(num_individuals)

        for m in range(self.num_objectives):
            sorted_front = sorted(front, key=lambda x: objectives[x][m])
            crowding_distance[0] = crowding_distance[num_individuals - 1] = float('inf')

            f_max = objectives[sorted_front[-1]][m]
            f_min = objectives[sorted_front[0]][m]

            if f_max == f_min:
                continue

            for i in range(1, num_individuals - 1):
                crowding_distance[i] += (objectives[sorted_front[i + 1]][m] - objectives[sorted_front[i - 1]][m]) / (f_max - f_min)

        return crowding_distance

    def select_parents(self, population, objectives):
        fronts, _, front_count = self.fast_nondominated_sort(population, objectives)
        selected_parents = []
        remaining = self.population_size
        i = 0

        while remaining > 0 and i < len(fronts):
            if remaining - front_count[i] >= 0:
                selected_parents.extend(fronts[i])
                remaining -= front_count[i]
            else:
                sorted_front = sorted(fronts[i], key=lambda x: self.crowding_distance_assignment(fronts[i], objectives)[x], reverse=True)
                selected_parents.extend(sorted_front[:remaining])
                remaining = 0
            i += 1

        return selected_parents

    def evolve(self, population, objectives):
        new_population = []

        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(self.select_parents(population, objectives), size=2, replace=False)
            child1, child2 = self.crossover(population[parent1], population[parent2])

            if np.random.rand() < self.mutation_rate:
                child1 = self.mutate(child1)
            if np.random.rand() < self.mutation_rate:
                child2 = self.mutate(child2)

            new_population.extend([child1, child2])

        return np.array(new_population)[:self.population_size]

    def optimize(self):
        population = self.initialize_population()
        objectives = np.random.rand(self.population_size, self.num_objectives)

        for _ in range(self.num_generations):
            population = self.evolve(population, objectives)

        return population, objectives

# Example usage
if __name__ == "__main__":
    nsga2 = NSGA2(population_size=100, num_generations=100, num_objectives=2, num_variables=5)
    final_population, final_objectives = nsga2.optimize()
    print("Final Population:")
    print(final_population)
    print("Final Objectives:")
    print(final_objectives)
