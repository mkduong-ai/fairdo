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
    pairs = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            pairs.append((lst[i], lst[j]))

    return pairs
