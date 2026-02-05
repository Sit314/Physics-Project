import numpy as np
import sympy as smp
from sympy.interactive import printing

printing.init_printing(use_latex=True)
from sympy.parsing.mathematica import parse_mathematica

# Chebishev matrix


def alt(n):
    alt = []
    for i in range(n):
        alt.append((-1) ** i)
    return np.array(alt)


def _alt(n):
    alt = []
    for i in range(n):
        alt.append((-1) ** (i + 1))
    return np.array(alt)


def consec(n):
    return list(range(n))


def cesnoc(n):
    cesnoc = []
    for i in range(n):
        cesnoc.append(-n + i)
    return cesnoc


def arrconsec(n):
    return np.array(list(range(n)))


def cheb(N):
    if N == 0:
        return 0, 1

    x = np.cos(np.pi * np.linspace(0, 1, N + 1))
    c = np.array([2] + [1] * (N - 1) + [2]) * alt(N + 1)
    X = np.outer(x, np.ones(N + 1))
    dX = X - X.T
    D = np.outer(c, np.array([1] * (N + 1)) / c) / (dX + np.identity(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x


def cheb_newdom(N, beta):
    D_N, u = cheb(N)
    return 2 * beta * D_N, (u + 1) / (2 * beta)


def fourier_dom(N, L):
    x = np.linspace(-L / 2, L / 2, N + 1)
    x = x[:-1]
    return x


def low_pass_filter(f, L):
    N = f.size  # Number of discretization points
    dx = L / N
    k = L * np.fft.fftfreq(N, d=dx)  # Define discrete wavenumbers
    f_hat = np.fft.fft(f)
    for i in range(len(k)):  # Filtering high frequencies
        if np.abs(k[i]) > (2 / 3) * k.max():
            f_hat[i] = 0.0
    fnew = np.real(np.fft.ifft(f_hat))
    return fnew


def low_pass_filter_with_u(f, L, u):
    f_low_pass_filter = f.copy()
    for u_i in range(len(u)):
        f_u_i = f[:, u_i]
        f_low_pass_filter[:, u_i] = low_pass_filter(f_u_i, L)
    return f_low_pass_filter


def fourier_derivative(f, L):
    N = f.size  # Number of discretization points
    x = fourier_dom(N, L)
    dx = L / N
    k = L * np.fft.fftfreq(N, d=dx)  # Define discrete wavenumbers
    f_hat = np.fft.fft(f)
    w_hat = (2 * np.pi / L) * k * f_hat * 1j
    dxf = np.real(np.fft.ifft(w_hat))
    return dxf


def fourier_derivative_two_variables(f, par):
    L, N_u = par  # the length of x segment and the number of grid points of u.
    N_x = f[:, 0].size  # Number of discretization points
    x = fourier_dom(N_x, L)
    dx = L / N_x
    k = L * np.fft.fftfreq(N_x, d=dx)  # Define discrete wavenumbers
    f_flat = f.flatten()
    k_flat = np.kron(np.ones(N_u + 1), k)
    f_flat_hat = np.fft.fft(f_flat)
    w_flat_hat = (2 * np.pi / L) * k_flat * f_flat_hat * 1j
    dxf_flat = np.real(np.fft.ifft(w_flat_hat))
    dxf = np.reshape(dxf_flat, (N_x, N_u + 1))
    return dxf


def readfiledouble(Y):
    file_path = r"C:\Users\matan\PycharmProjects\Black_brane_steady_state_stabilized_Horizon_Fixed_lambda\Yequations_Horizon_fixed_no_lambda.txt"
    file_path = r"Yequations_Horizon_fixed_no_lambda.txt"

    # Open and read the file
    with open(file_path, "r") as file:
        lines = file.readlines()
    for i in range(len(lines)):
        if lines[i].strip() == f"A21{Y}":
            A_21_r = lines[i + 1].strip()
        if lines[i].strip() == f"A22{Y}":
            A_22_r = lines[i + 1].strip()
        if lines[i].strip() == f"G2{Y}":
            G_2_r = lines[i + 1].strip()

    A_21 = parse_mathematica(A_21_r)
    A_22 = parse_mathematica(A_22_r)
    g_2 = parse_mathematica(G_2_r)
    return A_21, A_22, g_2


def readfilesingle(Y):
    file_path = r"C:\Users\matan\PycharmProjects\Black_brane_steady_state_stabilized_Horizon_Fixed_lambda\Yequations_Horizon_fixed_no_lambda.txt"
    file_path = r"Yequations_Horizon_fixed_no_lambda.txt"

    # Open and read the file
    with open(file_path, "r") as file:
        lines = file.readlines()
    for i in range(len(lines)):
        if lines[i].strip() == f"A{Y}":
            A_r = lines[i + 1].strip()
        if lines[i].strip() == f"G{Y}":
            G_r = lines[i + 1].strip()

    A = parse_mathematica(A_r)
    G = parse_mathematica(G_r)
    return A, G


# Equations:

# Solving for Sigma with y_1 and y_2:


def solve_y1_y2(known_func, N_beta_space, read_files):
    # Get the data
    a, f, b_n, lam = known_func
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    A_21, A_22, g_2 = read_files

    # Needed functions
    Super_D_N_u = np.kron(np.identity(2), D_N_u)

    # Chebishev derivatives:
    dub_n = np.matmul(D_N_u, b_n.T).T

    # Converting the txt from mathematica to numpy
    u, b, Dub = smp.symbols("u b Dub")
    A_21_f = smp.lambdify([u, b, Dub], A_21)
    A_22_f = smp.lambdify([u, b, Dub], A_22)
    g_2_f = smp.lambdify([u, b, Dub], g_2)

    # Solving the equations for every x3
    y_1_recover, y_2_recover = np.zeros((N_x3, N_u + 1)), np.zeros((N_x3, N_u + 1))
    for i_3 in range(N_x3):
        A_21_n = A_21_f(
            u_n_reg, b_n[i_3, :], dub_n[i_3, :]
        )  # We actually evaluate A_21(u,duB(u)), show Amos the kindergarten background file...
        A_22_n = A_22_f(u_n_reg, b_n[i_3, :], dub_n[i_3, :])
        g_2_n = g_2_f(u_n_reg, b_n[i_3, :], dub_n[i_3, :])
        Super_A = (
            np.kron(np.array([[0, 0], [1, 0]]), np.diag(A_21_n))
            + np.kron(np.array([[0, 1], [0, 0]]), np.identity(N_u + 1))
            + np.kron(np.array([[0, 0], [0, 1]]), np.diag(A_22_n))
        )
        Super_G = np.array([np.concatenate((np.zeros(g_2_n.size, dtype=float), g_2_n))]).T
        H = Super_D_N_u - Super_A
        y_1_vec = np.zeros((N_u + 1, 1))
        y_2_vec = np.zeros((N_u + 1, 1))
        y_vec = np.reshape(np.concatenate((y_1_vec, y_2_vec), axis=None), (2 * (N_u + 1), 1))
        S = Super_G - np.matmul(H, y_vec)

        new_line_1 = np.zeros((2 * N_u + 2), dtype=float)
        new_line_1[N_u] = 1.0
        new_line_2 = np.zeros((2 * N_u + 2), dtype=float)
        new_line_2[-1] = 1.0

        H[N_u, :] = new_line_1
        S[N_u, 0] = 0.0
        H[-1, :] = new_line_2
        S[-1, 0] = 0.0

        g = np.linalg.solve(H, S)
        y = g + y_vec
        y = y[:, 0]
        y_1 = y[: (N_u + 1)]
        y_2 = y[(N_u + 1) :]
        y_1_recover[i_3, :], y_2_recover[i_3, :] = y_1, y_2
    return y_1_recover, y_2_recover


# Solving for F_3 with y_3 and y_4:


def solve_y3_y4(known_func, N_beta_space, read_files):
    # Get the data
    a, f, b_n, lam = known_func
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    A_21_y1y2, A_22_y1y2, g_2_y1y2, A_21, A_22, g_2 = read_files
    read_files_old = A_21_y1y2, A_22_y1y2, g_2_y1y2

    # Needed functions
    Super_D_N_u = np.kron(np.identity(2), D_N_u)
    y_1_n, y_2_n = solve_y1_y2(known_func, N_beta_space, read_files_old)

    # Chebishev derivatives:
    dub_n = np.matmul(D_N_u, b_n.T).T
    dudub_n = np.matmul(D_N_u, dub_n.T).T
    duy_1_n = np.matmul(D_N_u, y_1_n.T).T
    duy_2_n = np.matmul(D_N_u, y_2_n.T).T

    # Fourier derivatives:
    par = beta_x3, N_u
    dx3b_n = fourier_derivative_two_variables(b_n, par)
    dx3dub_n = fourier_derivative_two_variables(dub_n, par)
    dx3y_1_n = fourier_derivative_two_variables(y_1_n, par)
    dx3duy_1_n = fourier_derivative_two_variables(duy_1_n, par)

    # Converting the txt from mathematica to numpy
    u, b, Dub, D3b, D3Dub, DuDub, y1, y2, D3y1, D3Duy1, Duy2 = smp.symbols(
        "u b Dub D3b D3Dub DuDub y1 y2 D3y1 D3Duy1 Duy2"
    )
    A_21_f = smp.lambdify([u, b, Dub, D3b, D3Dub, DuDub, y1, y2, D3y1, D3Duy1, Duy2], A_21)
    A_22_f = smp.lambdify([u, b, Dub, D3b, D3Dub, DuDub, y1, y2, D3y1, D3Duy1, Duy2], A_22)
    g_2_f = smp.lambdify([u, b, Dub, D3b, D3Dub, DuDub, y1, y2, D3y1, D3Duy1, Duy2], g_2)

    # Solving the equations for every x3
    y_3_recover, y_4_recover = np.zeros((N_x3, N_u + 1)), np.zeros((N_x3, N_u + 1))
    for i_3 in range(N_x3):
        A_21_n = A_21_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            duy_2_n[i_3, :],
        )  # We actually evaluate A_21(u,duB(u)), show Amos the kindergarten background file...
        A_22_n = A_22_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            duy_2_n[i_3, :],
        )
        g_2_n = g_2_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            duy_2_n[i_3, :],
        )
        Super_A = (
            np.kron(np.array([[0, 0], [1, 0]]), np.diag(A_21_n))
            + np.kron(np.array([[0, 1], [0, 0]]), np.identity(N_u + 1))
            + np.kron(np.array([[0, 0], [0, 1]]), np.diag(A_22_n))
        )
        Super_G = np.array([np.concatenate((np.zeros(g_2_n.size, dtype=float), g_2_n))]).T
        H = Super_D_N_u - Super_A
        y_3_vec = np.zeros((N_u + 1, 1))
        y_4_vec = np.array([f[i_3, :]]).T
        y_vec = np.reshape(np.concatenate((y_3_vec, y_4_vec), axis=None), (2 * (N_u + 1), 1))
        S = Super_G - np.matmul(H, y_vec)

        new_line_1 = np.zeros((2 * N_u + 2), dtype=float)
        new_line_1[N_u] = 1.0
        new_line_2 = np.zeros((2 * N_u + 2), dtype=float)
        new_line_2[-1] = 1.0

        H[N_u, :] = new_line_1
        S[N_u, 0] = 0.0
        H[-1, :] = new_line_2
        S[-1, 0] = 0.0

        g = np.linalg.solve(H, S)
        y = g + y_vec
        y = y[:, 0]
        y_3 = y[: (N_u + 1)]
        y_4 = y[(N_u + 1) :]
        y_3_recover[i_3, :], y_4_recover[i_3, :] = y_3, y_4
    return y_1_n, y_2_n, y_3_recover, y_4_recover


# Solving for d+Sigma with y_5:


def solve_y5(known_func, N_beta_space, read_files):
    # Get the data
    a, f, b_n, lam = known_func
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    A_21_y1y2, A_22_y1y2, g_2_y1y2, A_21_y3y4, A_22_y3y4, g_2_y3y4, A, g = read_files
    read_files_old = A_21_y1y2, A_22_y1y2, g_2_y1y2, A_21_y3y4, A_22_y3y4, g_2_y3y4

    # Loading the previous y's:
    y_1_n, y_2_n, y_3_n, y_4_n = solve_y3_y4(known_func, N_beta_space, read_files_old)

    # Chebishev derivatives:
    dub_n = np.matmul(D_N_u, b_n.T).T
    dudub_n = np.matmul(D_N_u, dub_n.T).T
    duy_1_n = np.matmul(D_N_u, y_1_n.T).T
    duy_2_n = np.matmul(D_N_u, y_2_n.T).T
    duy_3_n = np.matmul(D_N_u, y_3_n.T).T
    duy_4_n = np.matmul(D_N_u, y_4_n.T).T

    # Fourier derivatives:
    par = beta_x3, N_u
    dx3b_n = fourier_derivative_two_variables(b_n, par)
    dx3dub_n = fourier_derivative_two_variables(dub_n, par)
    dx3dx3b_n = fourier_derivative_two_variables(dx3b_n, par)
    dx3y_1_n = fourier_derivative_two_variables(y_1_n, par)
    dx3dx3y_1_n = fourier_derivative_two_variables(dx3y_1_n, par)
    dx3duy_1_n = fourier_derivative_two_variables(duy_1_n, par)
    dx3y_3_n = fourier_derivative_two_variables(y_3_n, par)
    dx3duy_3_n = fourier_derivative_two_variables(duy_3_n, par)

    # Converting the txt from mathematica to numpy
    u, b, Dub, D3b, D3Dub, DuDub, D3D3b = smp.symbols("u b Dub D3b D3Dub DuDub D3D3b")
    y1, y2, y3, y4, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, D3Duy3, Duy4 = smp.symbols(
        "y1 y2 y3 y4 D3y1 D3Duy1 D3D3y1 Duy2 D3y3 D3Duy3 Duy4"
    )
    A_f = smp.lambdify(
        [u, b, Dub, D3b, D3Dub, DuDub, D3D3b, y1, y2, y3, y4, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, D3Duy3, Duy4], A
    )
    g_f = smp.lambdify(
        [u, b, Dub, D3b, D3Dub, DuDub, D3D3b, y1, y2, y3, y4, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, D3Duy3, Duy4], g
    )

    # Solving the equations for every x3
    y_5_recover = np.zeros((N_x3, N_u + 1))
    for i_3 in range(N_x3):
        A_n = A_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            dx3dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            dx3dx3y_1_n[i_3, :],
            duy_2_n[i_3, :],
            dx3y_3_n[i_3, :],
            dx3duy_3_n[i_3, :],
            duy_4_n[i_3, :],
        )  # We actually evaluate A_21(u,duB(u)), show Amos the kindergarten background file...
        g_n = g_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            dx3dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            dx3dx3y_1_n[i_3, :],
            duy_2_n[i_3, :],
            dx3y_3_n[i_3, :],
            dx3duy_3_n[i_3, :],
            duy_4_n[i_3, :],
        )
        A_N, g_N = np.diag(A_n), np.array([g_n]).T

        H = D_N_u - A_N
        y_5_vec = np.array([a[i_3, :]]).T
        S = g_N - np.matmul(H, y_5_vec)

        new_line = np.zeros((N_u + 1), dtype=float)
        new_line[-1] = 1.0

        H[-1, :] = new_line
        S[-1, 0] = 0.0

        G = np.linalg.solve(H, S)
        y_5 = G + y_5_vec
        y_5_recover[i_3, :] = y_5.T[0]
    return y_1_n, y_2_n, y_3_n, y_4_n, y_5_recover


# Solving for d+B with y_6:


def solve_y6(known_func, N_beta_space, read_files):
    # Get the data
    a, f, b_n, lam = known_func
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    A_21_y1y2, A_22_y1y2, g_2_y1y2, A_21_y3y4, A_22_y3y4, g_2_y3y4, A_y5, g_y5, A_y6, g_y6 = read_files
    read_files_old = A_21_y1y2, A_22_y1y2, g_2_y1y2, A_21_y3y4, A_22_y3y4, g_2_y3y4, A_y5, g_y5

    # Needed functions
    b_D = np.matmul(D_N_u, b_n.T).T[:, -1]

    # Loading the previous y's:
    y_1_n, y_2_n, y_3_n, y_4_n, y_5_n = solve_y5(known_func, N_beta_space, read_files_old)

    # Chebishev derivatives:
    dub_n = np.matmul(D_N_u, b_n.T).T
    dudub_n = np.matmul(D_N_u, dub_n.T).T
    duy_1_n = np.matmul(D_N_u, y_1_n.T).T
    duy_2_n = np.matmul(D_N_u, y_2_n.T).T
    duy_3_n = np.matmul(D_N_u, y_3_n.T).T

    # Fourier derivatives:
    par = beta_x3, N_u
    dx3b_n = fourier_derivative_two_variables(b_n, par)
    dx3dub_n = fourier_derivative_two_variables(dub_n, par)
    dx3dx3b_n = fourier_derivative_two_variables(dx3b_n, par)
    dx3y_1_n = fourier_derivative_two_variables(y_1_n, par)
    dx3dx3y_1_n = fourier_derivative_two_variables(dx3y_1_n, par)
    dx3duy_1_n = fourier_derivative_two_variables(duy_1_n, par)
    dx3y_3_n = fourier_derivative_two_variables(y_3_n, par)
    dx3duy_3_n = fourier_derivative_two_variables(duy_3_n, par)

    # Converting the txt from mathematica to numpy
    u, b, Dub, D3b, D3Dub, DuDub, D3D3b = smp.symbols("u b Dub D3b D3Dub DuDub D3D3b")
    y1, y2, y3, y4, y5, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, D3Duy3 = smp.symbols(
        "y1 y2 y3 y4 y5 D3y1 D3Duy1 D3D3y1 Duy2 D3y3 D3Duy3"
    )
    A_f = smp.lambdify(
        [u, b, Dub, D3b, D3Dub, DuDub, D3D3b, y1, y2, y3, y4, y5, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, D3Duy3], A_y6
    )
    g_f = smp.lambdify(
        [u, b, Dub, D3b, D3Dub, DuDub, D3D3b, y1, y2, y3, y4, y5, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, D3Duy3], g_y6
    )

    # Solving the equations for every x3
    y_6_recover = np.zeros((N_x3, N_u + 1))
    for i_3 in range(N_x3):
        A_n = A_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            dx3dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            y_5_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            dx3dx3y_1_n[i_3, :],
            duy_2_n[i_3, :],
            dx3y_3_n[i_3, :],
            dx3duy_3_n[i_3, :],
        )  # We actually evaluate A_21(u,duB(u)), show Amos the kindergarten background file...
        g_n = g_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            dx3dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            y_5_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            dx3dx3y_1_n[i_3, :],
            duy_2_n[i_3, :],
            dx3y_3_n[i_3, :],
            dx3duy_3_n[i_3, :],
        )

        A_N, g_N = np.diag(A_n), np.array([g_n]).T

        H = D_N_u - A_N
        y_6_vec = -2 * b_D[i_3] * np.ones((N_u + 1, 1))
        S = g_N - np.matmul(H, y_6_vec)

        new_line = np.zeros((N_u + 1), dtype=float)
        new_line[-1] = 1.0

        H[-1, :] = new_line
        S[-1, 0] = 0.0

        G = np.linalg.solve(H, S)
        y_6 = y_6_vec + G
        y_6_recover[i_3, :] = y_6.T[0]
    return y_1_n, y_2_n, y_3_n, y_4_n, y_5_n, y_6_recover


# solving for A with y_7 and y_8:


def solve_y7_y8(known_func, N_beta_space, read_files):
    # Get the data
    a, f, b_n, lam = known_func
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    A_21_y1y2, A_22_y1y2, g_2_y1y2, A_21_y3y4, A_22_y3y4, g_2_y3y4, A_y5, g_y5, A_y6, g_y6, A_21, A_22, g_2 = read_files
    read_files_old = A_21_y1y2, A_22_y1y2, g_2_y1y2, A_21_y3y4, A_22_y3y4, g_2_y3y4, A_y5, g_y5, A_y6, g_y6

    # Needed functions
    Super_D_N_u = np.kron(np.identity(2), D_N_u)

    # Loading the previous y's:
    y_1_n, y_2_n, y_3_n, y_4_n, y_5_n, y_6_n = solve_y6(known_func, N_beta_space, read_files_old)

    # Chebishev derivatives:
    dub_n = np.matmul(D_N_u, b_n.T).T
    dudub_n = np.matmul(D_N_u, dub_n.T).T
    duy_1_n = np.matmul(D_N_u, y_1_n.T).T
    duy_2_n = np.matmul(D_N_u, y_2_n.T).T
    duy_4_n = np.matmul(D_N_u, y_4_n.T).T

    # Fourier derivatives:
    par = beta_x3, N_u
    dx3b_n = fourier_derivative_two_variables(b_n, par)
    dx3dub_n = fourier_derivative_two_variables(dub_n, par)
    dx3dx3b_n = fourier_derivative_two_variables(dx3b_n, par)
    dx3y_1_n = fourier_derivative_two_variables(y_1_n, par)
    dx3dx3y_1_n = fourier_derivative_two_variables(dx3y_1_n, par)
    dx3duy_1_n = fourier_derivative_two_variables(duy_1_n, par)
    dx3y_3_n = fourier_derivative_two_variables(y_3_n, par)

    # Converting the txt from mathematica to numpy
    u, b, Dub, D3b, D3Dub, DuDub, D3D3b = smp.symbols("u b Dub D3b D3Dub DuDub D3D3b")
    y1, y2, y3, y4, y5, y6, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, Duy4 = smp.symbols(
        "y1 y2 y3 y4 y5 y6 D3y1 D3Duy1 D3D3y1 Duy2 D3y3 Duy4"
    )
    A_21_f = smp.lambdify(
        [u, b, Dub, D3b, D3Dub, DuDub, D3D3b, y1, y2, y3, y4, y5, y6, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, Duy4], A_21
    )
    A_22_f = smp.lambdify(
        [u, b, Dub, D3b, D3Dub, DuDub, D3D3b, y1, y2, y3, y4, y5, y6, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, Duy4], A_22
    )
    g_2_f = smp.lambdify(
        [u, b, Dub, D3b, D3Dub, DuDub, D3D3b, y1, y2, y3, y4, y5, y6, D3y1, D3Duy1, D3D3y1, Duy2, D3y3, Duy4], g_2
    )

    # Solving the equations for every x3
    y_7_recover, y_8_recover = np.zeros((N_x3, N_u + 1)), np.zeros((N_x3, N_u + 1))
    for i_3 in range(N_x3):
        A_21_n = A_21_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            dx3dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            y_5_n[i_3, :],
            y_6_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            dx3dx3y_1_n[i_3, :],
            duy_2_n[i_3, :],
            dx3y_3_n[i_3, :],
            duy_4_n[i_3, :],
        )  # We actually evaluate A_21(u,duB(u)), show Amos the kindergarten background file...
        A_22_n = A_22_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            dx3dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            y_5_n[i_3, :],
            y_6_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            dx3dx3y_1_n[i_3, :],
            duy_2_n[i_3, :],
            dx3y_3_n[i_3, :],
            duy_4_n[i_3, :],
        )
        g_2_n = g_2_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            dx3dub_n[i_3, :],
            dudub_n[i_3, :],
            dx3dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            y_5_n[i_3, :],
            y_6_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3duy_1_n[i_3, :],
            dx3dx3y_1_n[i_3, :],
            duy_2_n[i_3, :],
            dx3y_3_n[i_3, :],
            duy_4_n[i_3, :],
        )
        if len(np.array([A_21_n])) == 1:
            A_21_n = np.zeros(N_u + 1)
        Super_A = (
            np.kron(np.array([[0, 0], [1, 0]]), np.diag(A_21_n))
            + np.kron(np.array([[0, 1], [0, 0]]), np.identity(N_u + 1))
            + np.kron(np.array([[0, 0], [0, 1]]), np.diag(A_22_n))
        )
        Super_G = np.array([np.concatenate((np.zeros(g_2_n.size, dtype=float), g_2_n))]).T
        H = Super_D_N_u - Super_A

        y_7_vec = np.zeros((N_u + 1, 1))
        y_8_vec = np.zeros((N_u + 1, 1))
        y_vec = np.reshape(np.concatenate((y_7_vec, y_8_vec), axis=None), (2 * (N_u + 1), 1))
        S = Super_G - np.matmul(H, y_vec)

        new_line_1 = np.zeros((2 * N_u + 2), dtype=float)
        new_line_1[N_u] = 1.0
        new_line_2 = np.zeros((2 * N_u + 2), dtype=float)
        new_line_2[-1] = 1.0

        H[N_u, :] = new_line_1
        S[N_u, 0] = 0.0
        H[-1, :] = new_line_2
        S[-1, 0] = 0.0

        g = np.linalg.solve(H, S)
        y = g + y_vec
        y = y[:, 0]
        y_7 = y[: (N_u + 1)]
        y_8 = y[(N_u + 1) :]
        y_7_recover[i_3, :], y_8_recover[i_3, :] = y_7, y_8
    return y_1_n, y_2_n, y_3_n, y_4_n, y_5_n, y_6_n, y_7_recover, y_8_recover


# Solving for d+F with y_9:


def solve_y9(known_func, N_beta_space, read_files):
    # Get the data
    a, f, b_n, lam = known_func
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    (
        A_21_y1y2,
        A_22_y1y2,
        g_2_y1y2,
        A_21_y3y4,
        A_22_y3y4,
        g_2_y3y4,
        A_y5,
        g_y5,
        A_y6,
        g_y6,
        A_21_y7y8,
        A_22_y7y8,
        g_2_y7y8,
        A_y9,
        g_y9,
    ) = read_files
    read_files_old = (
        A_21_y1y2,
        A_22_y1y2,
        g_2_y1y2,
        A_21_y3y4,
        A_22_y3y4,
        g_2_y3y4,
        A_y5,
        g_y5,
        A_y6,
        g_y6,
        A_21_y7y8,
        A_22_y7y8,
        g_2_y7y8,
    )

    # Loading the previous y's:
    y_1_n, y_2_n, y_3_n, y_4_n, y_5_n, y_6_n, y_7_n, y_8_n = solve_y7_y8(known_func, N_beta_space, read_files_old)

    # Chebishev derivatives:
    dub_n = np.matmul(D_N_u, b_n.T).T
    duy_5_n = np.matmul(D_N_u, y_5_n.T).T
    duy_6_n = np.matmul(D_N_u, y_6_n.T).T
    duy_7_n = np.matmul(D_N_u, y_7_n.T).T
    duy_8_n = np.matmul(D_N_u, y_8_n.T).T

    # Fourier derivatives:
    par = beta_x3, N_u
    dx3b_n = fourier_derivative_two_variables(b_n, par)
    dx3y_1_n = fourier_derivative_two_variables(y_1_n, par)
    dx3y_5_n = fourier_derivative_two_variables(y_5_n, par)
    dx3y_6_n = fourier_derivative_two_variables(y_6_n, par)
    dx3y_7_n = fourier_derivative_two_variables(y_7_n, par)
    dx3duy_7_n = fourier_derivative_two_variables(duy_7_n, par)

    # Converting the txt from mathematica to numpy
    u, b, Dub, D3b = smp.symbols("u b Dub D3b")
    y1, y2, y3, y4, y5, y6, y7, y8, D3y1, D3y5, D3y6, D3y7, D3Duy7, Duy5, Duy6, Duy7, Duy8 = smp.symbols(
        "y1 y2 y3 y4 y5 y6 y7 y8 D3y1 D3y5 D3y6 D3y7 D3Duy7 Duy5 Duy6 Duy7 Duy8"
    )
    A_f = smp.lambdify(
        [u, b, Dub, D3b, y1, y2, y3, y4, y5, y6, y7, y8, D3y1, D3y5, D3y6, D3y7, D3Duy7, Duy5, Duy6, Duy7, Duy8], A_y9
    )
    g_f = smp.lambdify(
        [u, b, Dub, D3b, y1, y2, y3, y4, y5, y6, y7, y8, D3y1, D3y5, D3y6, D3y7, D3Duy7, Duy5, Duy6, Duy7, Duy8], g_y9
    )

    # Solving the equations for every x3
    y_9_recover = np.zeros((N_x3, N_u + 1))
    for i_3 in range(N_x3):
        A_n = A_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            y_5_n[i_3, :],
            y_6_n[i_3, :],
            y_7_n[i_3, :],
            y_8_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3y_5_n[i_3, :],
            dx3y_6_n[i_3, :],
            dx3y_7_n[i_3, :],
            dx3duy_7_n[i_3, :],
            duy_5_n[i_3, :],
            duy_6_n[i_3, :],
            duy_7_n[i_3, :],
            duy_8_n[i_3, :],
        )  # We actually evaluate A_21(u,duB(u)), show Amos the kindergarten background file...
        g_n = g_f(
            u_n_reg,
            b_n[i_3, :],
            dub_n[i_3, :],
            dx3b_n[i_3, :],
            y_1_n[i_3, :],
            y_2_n[i_3, :],
            y_3_n[i_3, :],
            y_4_n[i_3, :],
            y_5_n[i_3, :],
            y_6_n[i_3, :],
            y_7_n[i_3, :],
            y_8_n[i_3, :],
            dx3y_1_n[i_3, :],
            dx3y_5_n[i_3, :],
            dx3y_6_n[i_3, :],
            dx3y_7_n[i_3, :],
            dx3duy_7_n[i_3, :],
            duy_5_n[i_3, :],
            duy_6_n[i_3, :],
            duy_7_n[i_3, :],
            duy_8_n[i_3, :],
        )

        A_N, g_N = np.diag(A_n), np.array([g_n]).T

        H = D_N_u - A_N
        y_9_vec = np.zeros((N_u + 1, 1))
        S = g_N - np.matmul(H, y_9_vec)

        new_line = np.zeros((N_u + 1), dtype=float)
        new_line[-1] = 1.0

        H[-1, :] = new_line
        S[-1, 0] = 0.0

        G = np.linalg.solve(H, S)
        y_9 = y_9_vec + G
        y_9_recover[i_3, :] = y_9.T[0]
    return y_1_n, y_2_n, y_3_n, y_4_n, y_5_n, y_6_n, y_7_n, y_8_n, y_9_recover


# Time propagation with RK4
# from parameters_CPS import Parameters
# N, D_N, u, beta, lam, dtlam = Parameters()


def fRK4(t, known_func, N_beta_space, read_files):
    a, f, b, lam = known_func
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    u_n = np.copy(u_n_reg)
    u_n[-1] = 0.0
    U, X3 = np.meshgrid(u_n, x3_n)

    # Low pass filter:
    # a = low_pass_filter_with_u(a_all,beta_x3,u_n)
    # f = low_pass_filter_with_u(f_all,beta_x3,u_n)
    # b = low_pass_filter_with_u(b_all,beta_x3,u_n)
    # lam = low_pass_filter_with_u(lam_all,beta_x3,u_n)

    # known_func = a, f, b, lam
    dub = np.matmul(D_N_u, b.T).T

    # Solving the set of ODE's
    y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8 = solve_y7_y8(known_func, N_beta_space, read_files)  # b_4 = dub[-1]
    # Set A_tilde, otherwise we will get divergences.
    dot_b_cut = y_6[:, :-1] + (y_7[:, :-1] * U[:, :-1] ** 2 + (1 / 2)) * (3 * b[:, :-1] / U[:, :-1] + dub[:, :-1])
    # dot_b = np.array([np.concatenate((dot_b_cut,np.array([0.])))]).T
    dot_b_zero = np.array([y_6[:, -1] + (y_7[:, -1] * U[:, -1] ** 2 + (1 / 2)) * (4 * dub[:, -1])]).T
    dot_b = np.insert(dot_b_cut, [len(dot_b_cut[0, :])], dot_b_zero, axis=1)

    b_D = np.kron(np.ones((N_u + 1, 1)), np.matmul(D_N_u, b.T).T[:, -1]).T
    par = beta_x3, N_u
    dot_a_4 = (2 / 3) * fourier_derivative_two_variables(f, par)
    dot_f3_4 = (1 / 2) * fourier_derivative_two_variables(a, par) + 2 * fourier_derivative_two_variables(b_D, par)

    # Low pass filter:
    # dot_b_filter = low_pass_filter_with_u(dot_b,beta_x3,u_n)
    # dot_a_4_filter = low_pass_filter_with_u(dot_a_4,beta_x3,u_n)
    # dot_f3_4_filter = low_pass_filter_with_u(dot_f3_4,beta_x3,u_n)
    # lam = low_pass_filter_with_u(lam_all,beta_x3,u_n)

    return dot_a_4, dot_f3_4, dot_b, 0.0


def kretschmann_scalar(Ys_and_b, dtY, dtdtY, N_beta_space, read_files):
    # Get the data
    y_1_n, y_2_n, y_3_n, y_4_n, y_5_n, y_6_n, y_7_n, y_8_n, y_9_n, b_n = Ys_and_b
    dtb_n, dty_1_n, dty_7_n = dtY
    dtdtb_n, dtdty_1_n = dtdtY
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    Rsquare = read_files

    # Needed functions
    # u_n = np.copy(u_n_reg)
    # u_n[-1] = 0.
    U_reg, X3_reg = np.meshgrid(u_n_reg, x3_n)

    # Chebishev derivatives:
    dub_n = np.matmul(D_N_u, b_n.T).T
    dudub_n = np.matmul(D_N_u, dub_n.T).T
    duy_1_n = np.matmul(D_N_u, y_1_n.T).T
    duy_2_n = np.matmul(D_N_u, y_2_n.T).T
    duy_3_n = np.matmul(D_N_u, y_3_n.T).T
    duy_4_n = np.matmul(D_N_u, y_4_n.T).T
    duy_5_n = np.matmul(D_N_u, y_5_n.T).T
    duy_6_n = np.matmul(D_N_u, y_6_n.T).T
    duy_7_n = np.matmul(D_N_u, y_7_n.T).T
    duy_8_n = np.matmul(D_N_u, y_8_n.T).T
    duy_9_n = np.matmul(D_N_u, y_9_n.T).T

    # Fourier derivatives:
    par = beta_x3, N_u
    dx3b_n = fourier_derivative_two_variables(b_n, par)
    dx3dub_n = fourier_derivative_two_variables(dub_n, par)
    dx3dx3b_n = fourier_derivative_two_variables(dx3b_n, par)
    dx3y_1_n = fourier_derivative_two_variables(y_1_n, par)
    dx3y_3_n = fourier_derivative_two_variables(y_3_n, par)
    dx3y_5_n = fourier_derivative_two_variables(y_5_n, par)
    dx3y_6_n = fourier_derivative_two_variables(y_6_n, par)
    dx3y_7_n = fourier_derivative_two_variables(y_7_n, par)
    dx3y_9_n = fourier_derivative_two_variables(y_9_n, par)
    dx3dx3y_1_n = fourier_derivative_two_variables(dx3y_1_n, par)
    dx3dx3y_7_n = fourier_derivative_two_variables(dx3y_7_n, par)
    dx3duy_1_n = fourier_derivative_two_variables(duy_1_n, par)
    dx3duy_3_n = fourier_derivative_two_variables(duy_3_n, par)
    dx3duy_7_n = fourier_derivative_two_variables(duy_7_n, par)

    # Converting the txt from mathematica to numpy
    u, b, Dub, D3b, D3Dub, DuDub, D3D3b = smp.symbols("u b Dub D3b D3Dub DuDub D3D3b")
    y1, y2, y3, y4, y5, y6, y7, y8, y9 = smp.symbols("y1 y2 y3 y4 y5 y6 y7 y8 y9")
    Duy1, Duy2, Duy3, Duy4, Duy5, Duy6, Duy7, Duy8, Duy9 = smp.symbols("Duy1 Duy2 Duy3 Duy4 Duy5 Duy6 Duy7 Duy8 Duy9")
    D3y1, D3y3, D3y5, D3y6, D3y7, D3y9 = smp.symbols("D3y1 D3y3 D3y5 D3y6 D3y7 D3y9")
    D3D3y1, D3D3y7, D3Duy1, D3Duy3, D3Duy7 = smp.symbols("D3D3y1 D3D3y7 D3Duy1 D3Duy3 D3Duy7")
    Dtb, Dty1, Dty7, DtDtb, DtDty1 = smp.symbols("Dtb Dty1 Dty7 DtDtb DtDty1")

    Rsquare_f = smp.lambdify(
        [
            u,
            b,
            Dub,
            D3b,
            D3Dub,
            DuDub,
            D3D3b,
            y1,
            y2,
            y3,
            y4,
            y5,
            y6,
            y7,
            y8,
            y9,
            Duy1,
            Duy2,
            Duy3,
            Duy4,
            Duy5,
            Duy6,
            Duy7,
            Duy8,
            Duy9,
            D3y1,
            D3y3,
            D3y5,
            D3y6,
            D3y7,
            D3y9,
            D3D3y1,
            D3D3y7,
            D3Duy1,
            D3Duy3,
            D3Duy7,
            Dtb,
            Dty1,
            Dty7,
            DtDtb,
            DtDty1,
        ],
        Rsquare,
    )

    Rsquare_n = Rsquare_f(
        u_n_reg,
        b_n,
        dub_n,
        dx3b_n,
        dx3dub_n,
        dudub_n,
        dx3dx3b_n,
        y_1_n,
        y_2_n,
        y_3_n,
        y_4_n,
        y_5_n,
        y_6_n,
        y_7_n,
        y_8_n,
        y_9_n,
        duy_1_n,
        duy_2_n,
        duy_3_n,
        duy_4_n,
        duy_5_n,
        duy_6_n,
        duy_7_n,
        duy_8_n,
        duy_9_n,
        dx3y_1_n,
        dx3y_3_n,
        dx3y_5_n,
        dx3y_6_n,
        dx3y_7_n,
        dx3y_9_n,
        dx3dx3y_1_n,
        dx3dx3y_7_n,
        dx3duy_1_n,
        dx3duy_3_n,
        dx3duy_7_n,
        dtb_n,
        dty_1_n,
        dty_7_n,
        dtdtb_n,
        dtdty_1_n,
    )

    return Rsquare_n


def Sigma(t, S, N):
    Y_1 = np.array([[] for i in range(N + 1)])
    Y_3 = np.array([[] for i in range(N + 1)])
    Y_4 = np.array([[] for i in range(N + 1)])
    Y_5 = np.array([[] for i in range(N + 1)])
    for n in range(len(t)):
        b_u_t_new, a_t_4new = S
        # b_4 = np.matmul(D_N4,B).T[0][-1]/24
        dub = np.matmul(D_N, np.array([b_u_t_new[:, n]]).T).T[0]
        b = b_u_t_new[:, n]
        duB = dub * u**3 + 3 * b * u**2
        a_4new = a_t_4new[0][n]
        b_4new = dub[-1]
        y_1, y_2, y_3, y_4, y_5, y_6 = solve_y5_y6(a_4new, b_4new, lam, dtlam, N, duB, beta)
        Y_1 = np.insert(Y_1, [len(Y_1[-1])], np.array([y_1]).T, axis=1)
        Y_3 = np.insert(Y_3, [len(Y_3[-1])], np.array([y_3]).T, axis=1)
        Y_4 = np.insert(Y_4, [len(Y_4[-1])], np.array([y_4]).T, axis=1)
        Y_5 = np.insert(Y_5, [len(Y_5[-1])], np.array([y_5]).T, axis=1)
    return Y_1, Y_3, Y_4, Y_5
