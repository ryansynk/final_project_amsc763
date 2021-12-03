import numpy as np

def house(x):
    e1 = np.zeros(x.size)
    e1[0] = 1
    v = x - np.exp(1j*np.angle(x[0])) * np.linalg.norm(x) * e1
    return np.atleast_2d(v).T, (2 / (np.vdot(v, v)))

def qr(A):
    m = A.shape[0]
    n = A.shape[1]
    Q = np.eye(m, dtype='complex')

    for k in range(n):
        v, beta = house(A[k:m,k])
        A[k:m, k:n] = A[k:m, k:n] - beta * v @ (v.conj().T @A[k:m, k:n])
        Q[:, k:m] = Q[:, k:m] - beta * (Q[:, k:m] @ v) @ v.conj().T

    return Q

def main():
    A = np.array([[0, 1], [1, 1], [1, 1], [2, 1]], dtype='complex')
    Q = qr(A)
    Q_true, r_true = np.linalg.qr(A)

    print(Q)
    print(Q_true)

if __name__ == "__main__":
    main()
