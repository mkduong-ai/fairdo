import numpy as np

from evaluation.nonbinary.optimize.Penalty import penalty
from evaluation.nonbinary.optimize.Penalty import penalty_normalized


# TODO: add constraint
# TODO: if not better, do not do anything
def metric_optimizer_remover_constraint(f, d, n_constraint,
                                        frac=0.75,
                                        m_cands=5,
                                        eps=0,
                                        penalty=penalty_normalized):
    """
    Parameters
    ----------
    f: callable
        function to optimize
    d: int
        number of dimensions
    n_constraint: int
        constraint of number of 1s
    frac: float
        A multiplicative or fraction of the given dataset's size
    m_cands: int
        number of candidates to select
    eps: float
        stop criterion. Returns solution if f(solution) <= eps.
    penalty: callable
        The penalty function that penalizes the fitness of a solution if it does not satisfy the constraint.
        Parameters: solution (np.array), n (int)
        Returns: float
    Returns
    -------
    solution: np.array of size (d,)
        The best solution found by the algorithm
    fitness: float
        The fitness of the best solution found by the algorithm
    """
    if frac < 1:
        n = int(d * (1-frac))
    else:
        raise Exception('Frac not valid. Fraction frac must be smaller than 1.')

    initial_solution = np.ones(d)
    solution = np.copy(initial_solution)
    fitness = 1 # assume 1 is worst fitness

    # check if dataset is already fair and return solution immediately
    if f(initial_solution) <= eps:
        return initial_solution, f(initial_solution)
    # removal loop
    for i in range(n):
        # choose m random indices that have a value 1
        idx = np.where(solution == 1)[0]
        cands_idx = np.random.choice(idx, m_cands, replace=False)
        # flip each bit once and save fitness
        scores = []
        for j in cands_idx:
            solution[j] = 1 - solution[j] # flip bit
            scores.append(f(solution))
            solution[j] = 1 - solution[j] # revert flip
        # select best candidate and flip bit
        best_cand = np.argmin(scores)
        solution[cands_idx[best_cand]] = 1 - solution[cands_idx[best_cand]]
        fitness = f(solution)
        # stop early if discrimination threshold satisfied
        if fitness <= eps:
            break

    return solution, fitness


def metric_optimizer_remover(f, d, m_cands):
    """
    Parameters
    ----------
    f: function
        function to optimize
    d: int
        number of dimensions
    m_cands: int
        number of candidates to select
    Returns
    -------
    solution: np.array of size (d,)
        The best solution found by the algorithm
    fitness: float
        The fitness of the best solution found by the algorithm
    """
    return metric_optimizer_remover_constraint(f=f, d=d, n_constraint=0, m_cands=m_cands)
