import numpy as np

def two_phase_simplex(c, A, b):
    m, n = A.shape
    c_hat = np.concatenate((np.zeros(n), np.ones(m)))
    A_hat = np.concatenate((A, np.eye(m)), axis=1)
    basis = list(range(n, n + m))
    
    # Phase 1
    while True:
        # Choose the entering variable
        j = np.argmin(c_hat)
        if c_hat[j] >= 0:
            break
        
        # Choose the leaving variable
        ratios = [b[i] / A_hat[i, j] if A_hat[i, j] > 0 else np.inf for i in range(m)]
        i = np.argmin(ratios)
        if ratios[i] == np.inf:
            raise Exception("Unbounded")
        
        # Update the basis
        basis[i] = j
        
        # Update the tableau
        pivot = A_hat[i, j]
        A_hat[i, :] /= pivot
        b[i] /= pivot
        for k in range(m):
            if k != i:
                factor = A_hat[k, j]
                A_hat[k, :] -= factor * A_hat[i, :]
                b[k] -= factor * b[i]
        c_hat -= c_hat[j] * A_hat[i, :]
    
    # Phase 2
    while True:
        # Calculate the reduced costs
        c_reduced = c - np.dot(c[basis], A)
        
        # Check if optimal
        if np.all(c_reduced >= 0):
            break
        
        # Choose the entering variable
        j = np.argmin(c_reduced)
        if c_reduced[j] >= 0:
            raise Exception("Unbounded")
        
        # Choose the leaving variable
        ratios = [b[i] / A[i, j] if A[i, j] > 0 else np.inf for i in range(m)]
        i = np.argmin(ratios)
        if ratios[i] == np.inf:
            raise Exception("Unbounded")
        
        # Update the basis
