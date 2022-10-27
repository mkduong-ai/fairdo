import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


def mutual_information(y: np.array, z: np.array, bins=2, **kwargs) -> float:
    """
    Measures whether two variables are independent by calculating their mutual information

    Parameters
    ----------
    y: np.array
    z: np.array
    bins: int
    kwargs: keyworded arguments

    Returns
    -------
    float
    """
    # discrimination measurement
    c_zy = np.histogram2d(z, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_zy)
    return mi


def normalized_mutual_information(y: np.array, z: np.array, **kwargs) -> float:
    """
    Returns the normalized mutual information between two variables.

    Parameters
    ----------
    y: np.array
    z: np.array
    kwargs: keyworded arguments

    Returns
    -------
    float
    """
    # Normalizes mutual information to 0 (independence) and 1 (perfect correlation)
    return normalized_mutual_info_score(y, z)


def separation():
    pass
    

def sufficiency():
    pass


def rdc(y: np.array, z: np.array, f=np.sin, k=20, s=1 / 6., n=1, **kwargs):
    """
    Implements the Randomized Dependence Coefficient
    David Lopez-Paz, Philipp Hennig, Bernhard Schoelkopf

    http://papers.nips.cc/paper/5138-the-randomized-dependence-coefficient.pdf
    Computes the Randomized Dependence Coefficient
    y, z: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)

    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    """
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(y, z, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(y.shape) == 1: y = y.reshape((-1, 1))
    if len(z.shape) == 1: z = z.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in y.T]) / float(y.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in z.T]) / float(z.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))
