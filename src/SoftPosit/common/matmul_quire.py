import softposit as sp
import numpy as np


def matmul_quire(a, b):
    # print("Posit_Quire multip.")
    assert a.ndim == b.ndim == 2
    t_a = type(a[0, 0])
    t_b = type(b[0, 0])

    if t_a == sp.posit8:
        posit_t = sp.posit8
        q = sp.quire8()
    # elif t_a == sp.posit16:
    #     posit_t = sp.posit16
    #     q = sp.quire16()
    # elif t_a == sp.posit32:
    #     posit_t = sp.posit32
    #     q = sp.quire32()
    else:
        raise TypeError("Only posit types from SoftPosit accepted. Received {} and {}".format(
            str(t_a), str(t_b)))

    ar, ac = a.shape  # n_rows * n_cols
    br, bc = b.shape  # n_rows * n_cols
    # rows == columns
    assert ac == br, "Matrix inner dimensions mismatch. A is {} and B is {}".format(
        str(ac), str(br))

    c = np.empty((ar, bc), dtype=posit_t)
    # c2 = np.zeros((ar, bc), dtype=posit_t)
    for i in range(ar):
        for j in range(bc):
            q.clr()
            for k in range(ac):  # or br
                # c[i,j] += a[i,k] * b[k,j]
                q.qma(a[i, k], b[k, j])
                # c2[i,j] += a[i,k] * b[k,j]
            c[i, j] = q.toPosit()
    return c


def matmul_no_quire(a, b):
    assert a.ndim == b.ndim == 2

    if type(a[0, 0]) == sp.posit8:
        posit_t = sp.posit8
    elif type(a[0, 0]) == sp.posit16:
        posit_t = sp.posit16
    elif type(a[0, 0]) == sp.posit32:
        posit_t = sp.posit32
    else:
        raise TypeError('Only posit types form SoftPosit accepted')

    ar, ac = a.shape  # n_rows * n_cols
    br, bc = b.shape  # n_rows * n_cols
    assert ac == br  # rows == columns

    c = np.zeros((ar, bc), dtype=posit_t)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):  # or br
                c[i, j] += a[i, k] * b[k, j]
    return c
