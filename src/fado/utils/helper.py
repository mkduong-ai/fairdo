from itertools import combinations


def generate_pairs(lst):
    """
    Generate all possible pairs of elements in a list without repetitions

    Parameters
    ----------
    lst: list or np.array
        list of elements

    Returns
    -------
    list of pairs
    """
    return list(combinations(lst, 2))
