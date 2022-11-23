
import taichi as ti
import time

ti.init(arch=ti.cuda, dynamic_index = True)

@ti.func
def sign(x: ti.template()):
    res = 1.0
    if x < 0:
        res = -1.0
    return res

@ti.func
def householder_vector_decomp(x: ti.template()):
    x2 = x[1:]
    X2 = 0.0
    if ti.static(x.n > 1):
        X2 = x2.norm() 
    alpha = x.norm()
    rho = -sign(x[0]) * alpha
    v1 = x[0] - rho
    u2 = x2 / v1
    X2 = X2 / ti.abs(v1)
    tau = (1 + X2 * X2) / 2

    u = ti.types.vector(u2.n + 1, ti.f32)(0)
    u[0] = 1
    for i in ti.static(range(1, u2.n + 1)):
        u[i] = u2[i - 1]
    return rho, u, tau

@ti.func
def householder_decomp(M: ti.template()):
    Q = ti.Matrix.identity(ti.f32, M.n)
    R = M
    # Proceeeds for all m columns
    for j in ti.static(range(M.m)):
        col = ti.types.vector(M.n - j, ti.f32)(0)
        for k in ti.static(range(M.n - j)):
            col[k] = R[k + j, j]
        rho, u, tau = householder_vector_decomp(col)
        Hk = ti.Matrix.identity(ti.f32, M.n)
        v = ti.types.vector(j + u.n, ti.f32)(0)
        for k in ti.static(range(u.n)):
            v[k + j] = u[k]
        Hk -= (1.0 / tau) * (v @ v.transpose())
        R = Hk @ R
        Q = Q @ Hk 

    return Q, R


@ti.func
def easy_solve(A: ti.template()):
    mat = A[:3, :3]
    b = A[:3, 3]
    b = mat.transpose() @ b
    mat = mat.transpose() @ mat
    U, Sigma, V = ti.svd(mat)
    for i in ti.static(range(3)):
        Sigma[i, i] = sign(Sigma[i, i]) * ti.max(0.1, ti.abs(Sigma[i, i]))
    pseudoInv = V @ Sigma.inverse() @ U.transpose()
    return  pseudoInv @ b
    


@ti.func
def solve_qef(A: ti.template()):
    # SVD decomposition
    U, Sigma, V = ti.svd(A[:3, :3])
    # Truncate result
    for i in ti.static(range(3)):
        Sigma[i, i] = sign(Sigma[i, i]) * ti.max(0.1, ti.abs(Sigma[i, i]))
    # Calculate pseudo-inverse matrix
    pseudoInv = V @ Sigma.inverse() @ U.transpose()
    b = A[:3, 3]
    # Return result
    return pseudoInv @ b


@ti.kernel
def testWrap():
    mat = ti.Matrix([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1]
    ])
    Q, R = householder_decomp(mat)
    print(Q @ R - mat)


@ti.kernel
def test_impl(num_epoch: ti.i32, n: ti.i32, m: ti.i32, eps: ti.f32) -> ti.i32:
    num_diff = 0
    for i in range(num_epoch):
        A = ti.types.matrix(3, 2, ti.f32)(0) 
        for x, y in ti.ndrange(3, 2):
            A[x, y] = ti.random() * 100

        A = ti.Matrix([
            [1, 1],
            [1, 1],
            [0, 3]
        ])

        Q, R = householder_decomp(A)
        B = Q @ R
        print(R)

        for x, y in ti.ndrange(3, 2):
            if ti.abs(A[x, y] - B[x, y]) > eps:
                num_diff += 1
                break
    return num_diff


def test(num_epoch,  eps = 0.0001):
    res = test_impl(num_epoch, 5, 4, 0.001)
    if res > 0:
        print("Failed test dimeion: {}x{}. Failed Count: {}".format(5, 4, res))
    else:
        print("QR householder test succeed!")

if __name__ == "__main__":
    # now = time.time()
    # test(10000000)
    # end = time.time()
    # print("Total used time: {}".format(end - now))
    # test(10000000)

    # ti.profiler.print_kernel_profiler_info()
    test(1)    
    # R = testWrap()

    
        









