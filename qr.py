import numpy as np
import time

"""
Return householder reflection vector
"""
def house(x):
    e1 = np.zeros(x.size)
    e1[0] = 1
    v = x - np.linalg.norm(x) * e1
    return np.atleast_2d(v).T, (2 / (np.dot(v, v)))

"""
Factor matrix A = QR. Q unitary, R upper triangular
"""
def qr(A):
    m = A.shape[0]
    n = A.shape[1]
    Q = np.eye(m)
    R = A.copy()

    for k in range(0, n):
        v, beta = house(R[k:,k])
        R[k:m, k:n] = R[k:m, k:n] - beta * v @ (v.T @ R[k:m, k:n])
        Q[:, k:m] = Q[:, k:m] - beta * (Q[:, k:m] @ v) @ v.T

    return Q, R

"""
For testing: generates a 2n x n random matrix
"""
def gen_matrix(n):
    return np.random.rand(2*n, n)

"""
Iterates over different sizes, checks that it works, times result
"""
def test_qr(n_max):
    times = []
    for i in range(n_max):
        n = 10*i
        A = gen_matrix(n)
        start = time.time()
        Q, R = qr(A)
        end = time.time()
        elapsed = end - start
        np.testing.assert_allclose(A, Q @ R, rtol = 1e-9, atol = 1e-9)
        print("PASS for problem size: " + str(A.shape) + " in time: " + str(elapsed))
        times.append(elapsed)

    return times
        

def main():
    times = test_qr(30)

if __name__ == "__main__":
    main()
