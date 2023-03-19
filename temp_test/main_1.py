import numpy as np

def simplex_method_2phase(c, A, b):
    # First, check if any of the b values are negative. If so, multiply the
    # corresponding row of A and b by -1 to make them positive.
    neg_b_indices = np.where(b < 0)[0]
    if len(neg_b_indices) > 0:
        A[neg_b_indices] *= -1
        b[neg_b_indices] *= -1

    # Initialize the auxiliary LP problem
    m, n = A.shape
    c_aux = np.zeros(n + m)
    c_aux[n:] = 1
    A_aux = np.zeros((m, n + m))
    A_aux[:, :n] = A
    A_aux[:, n:] = np.eye(m)
    b_aux = b.copy()

    # Phase 1
    B_indices = np.arange(n, n + m)
    N_indices = np.arange(n)
    c_bar = np.concatenate((np.zeros(n), np.ones(m)))
    A_bar = A_aux.copy()
    v = 0  # Initialize the objective value to 0

    while True:
        # Choose entering variable
        entering_index = np.argmax(c_bar)
        if c_bar[entering_index] == 0:
            # If all c_bar values are non-positive, we have found an initial BFS
            break

        # Choose leaving variable using Bland's rule
        candidates = np.where(A_bar[:, entering_index] > 0)[0]
        ratios = b_aux[candidates] / A_bar[candidates, entering_index]
        leaving_index = candidates[np.argmin(ratios)]
        B_indices = np.delete(B_indices, np.where(B_indices == leaving_index))
        N_indices = np.delete(N_indices, np.where(N_indices == entering_index))
        B_indices = np.concatenate((B_indices, np.array([entering_index])))
        N_indices = np.concatenate((N_indices, np.array([leaving_index])))

        # Update the tableau
        pivot = A_bar[leaving_index, entering_index]
        A_bar[leaving_index, :] /= pivot
        b_aux[leaving_index] /= pivot
        for i in range(m):
            if i != leaving_index:
                multiplier = A_bar[i, entering_index]
                A_bar[i, :] -= multiplier * A_bar[leaving_index, :]
                b_aux[i] -= multiplier * b_aux[leaving_index]
