# -*- coding: UTF-8 -*-

import numpy as np

from pyspark.mllib.linalg import SparseVector

def minhash(v, a, b, p, m):
    """
    Determines the type and computes the minhash of the vector.
        1: Multiplies the index by the non-zero seed "a".
        2: Adds the bias "b" (can be 0).
        3: Modulo "p", a number larger than the number of elements.
        4: Modulo "m", the number of buckets.

    Parameters
    ----------
    v : object
        Python list, NumPy array, or SparseVector.
    a : integer
        Seed, > 0.
    b : integer
        Seed, >= 0.
    p : integer
        Only restriction is that this number is larger than the number of elements.
    m : integer
        Number of bins.

    Returns
    -------
    i : integer
        Integer minhash value that is in [0, buckets).
    """
    indices = None
    
    if type(v) is SparseVector:
        indices = v.indices  
    elif type(v) is np.ndarray or type(v) is list:
        indices = np.arange(len(v), dtype = np.int)
        indices = indices[v == True]
    else:
        raise Exception("Unknown array type '%s'." % type(v))
        
    # Map the indices to hash values and take the minimum.
    return ((((a * indices) + b) % p) % m).min()