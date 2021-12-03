import numpy as np

def house(x):
    e1 = np.zeros(x.size)
    e1[0] = 1
    v = x - np.linalg.norm(x) * e1
    return np.atleast_2d(v).T, (2 / (np.dot(v, v)))

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

def main():
    A = np.array([[0, 1], [1, 1], [1, 1], [2, 1]], dtype='float')
    Q, R  = qr(A)
    Q_true, R_true = np.linalg.qr(A, mode='complete')
    assert(np.linalg.norm(Q @ R - A) <= 1e-15)

if __name__ == "__main__":
    main()
