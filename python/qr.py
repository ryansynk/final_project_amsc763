import numpy as np
import time
import argparse
import logging
import sys

"""
Return householder reflection vector
"""
def house(x):
    e1 = np.zeros(x.size)
    e1[0] = 1
    v = x - np.linalg.norm(x) * e1
    if np.all(v == np.zeros(v.shape)):
        return np.atleast_2d(np.zeros(v.shape)).T, 0
    else:
        return np.atleast_2d(v).T, (2 / (np.dot(v, v)))

"""
Factor matrix A = QR. Q unitary, R upper triangular
r = size of block, n / r is number of blocks
"""
def block_qr(A, r):
    np.set_printoptions(precision=6, suppress=True,linewidth=sys.maxsize)
    m = A.shape[0]
    n = A.shape[1]
    Q = np.eye(m)
    R = A.copy()
    assert(n % r == 0) # Assumes n divisible by r for simplicity

    B = np.zeros(r)

    for k in range(0, int(n / r)): # Loops over all blocks
        s = k * r
        V = np.zeros([m - s, r])
        for j in range(0, r): # Loops over every column in a block
            u = s + j
            v, beta = house(R[u:, u])
            R[u:m, u:(s + r)] = R[u:m, u:(s + r)] - beta * v @ (v.T @ R[u:m, u:(s + r)])
            v_zero_pad = np.zeros([m - s, 1])
            v_zero_pad[(m - s - v.shape[0]):m, :] = v
            V[:, j] = v_zero_pad[:, 0]
            B[j] = beta

        # Generate matrices W, Y such that P = I - W @ Y.T
        Y = np.atleast_2d(V[:,0]).T
        W = -B[0] * np.atleast_2d(V[:, 0]).T
        for j in range(1, r):
            v = np.atleast_2d(V[:, j]).T
            z = -B[j] * v - B[j] * W @ (Y.T @ v)
            W = np.hstack((W, z))
            Y = np.hstack((Y, v))

        # Update Q, R
        R[s:, s + r:] = R[s:, s + r:] + Y @ (W.T @ R[s:, s + r:])
        Q[:, s:] = Q[:, s:] + Q[:, s:] @ W  @ Y.T

    return Q, R

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
    for i in range(1, n_max):
        n = 10*i
        A = gen_matrix(n)
        start = time.time()
        Q, R = qr(A)
        end = time.time()
        elapsed = end - start
        np.testing.assert_allclose(A, Q @ R, rtol = 1e-9, atol = 1e-9)
        print("qr PASS for problem size: " + str(A.shape) + " in: " + str(1000*elapsed) + " milliseconds")
        times.append(elapsed)

    return times

def test_block_qr(n_max, r):
    times = []
    for i in range(1, n_max):
        #n = 30*i
        n = i * 64 * 5
        A = gen_matrix(n)
        start = time.time()
        Q, R = block_qr(A, r)
        end = time.time()

        np.testing.assert_allclose(A, Q @ R, rtol = 1e-7, atol = 1e-7)
        elapsed = end - start
        print("block_qr PASS. r = " + str(r)  + ". (m, n) = " + str(A.shape) + ". time = " + str(elapsed))
        times.append(elapsed)

    return times

def deterministic_test_block_qr(A, r):
    start = time.time()
    Q, R = block_qr(A, r)
    end = time.time()
    Q_true, R_true = np.linalg.qr(A, mode='complete')
    np.testing.assert_allclose(Q, Q_true, rtol = 1e-7, atol = 1e-7)
    np.testing.assert_allclose(R, R_true, rtol = 1e-7, atol = 1e-7)
    elapsed = end - start
    print("Deterministic block_qr PASS. r = " + str(r)  + ". (m, n) = " + str(A.shape) + ". time = " + str(elapsed))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", dest='debug', default=False, action='store_true')
    args = parser.parse_args()
    if (args.debug): 
        #logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
        logging.basicConfig(stream=sys.stderr)
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--debug")
    #args = parser.parse_args()
    #qr_times = test_qr(30)
    #block_qr_times = test_block_qr(11, r=3)

    #A = np.array([[ 0.521103,  0.251159,  0.448416 ],
    #              [-0.557204,  0.614949, -0.261701 ],
    #              [ 0.561759,  0.781652, -0.327026 ],
    #              [ 0.461415, -0.611471,  0.422656 ],
    #              [ 0.073847,  0.190156, -0.787745 ],
    #              [-0.233277, -0.650450, -0.508967 ]])
    #Q, R = qr(A)
    #print("Q = ")
    #print(Q)
    #print("R = ")
    #print(R)

    #A = np.array([[0.8054398 , 0.11770048, 0.74435746, 0.07596747],
    #              [0.47612782, 0.95610043, 0.91532087, 0.73867671],
    #              [0.43006959, 0.61098952, 0.2653968 , 0.61539964],
    #              [0.90222967, 0.13762961, 0.24488956, 0.57760962],
    #              [0.08671578, 0.33511532, 0.13160944, 0.7750951 ],
    #              [0.63046399, 0.96516845, 0.95523958, 0.99198526],
    #              [0.34393792, 0.18000136, 0.95844227, 0.39069116],
    #              [0.71946612, 0.91549769, 0.6170415 , 0.35973015]])

    #qr_times = test_qr(30)
    block_qr_times = test_block_qr(14, r=64)
    #deterministic_test_block_qr(A, 2)

    #A = np.loadtxt("A_matrix_8_by_4")
    #block_qr(A, 4)


if __name__ == "__main__":
    main()
