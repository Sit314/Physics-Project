import numpy as np
import math


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
            f_hat[i] = 0.
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

# Fixed it! no extrapolations are allowed
# def b_lambda_transformation_fixed_len_two_var(u_n, len_x, N_interpolation, b_prev, lam_new):
#     b_new = np.zeros(b_prev.shape)
#     min_lam = lam_new[:,0].min()
#     if min_lam < 0:
#         min_lam = 0.
#     for i_3 in range(len_x):
#         u_cut = u_n.max()/(1+min_lam*u_n.max()) #lambda is independant on u.
#         D_N_uu, uu = cheb_newdom(u_n.size-1, 1/u_cut)
#         poly_coeff = np.polynomial.chebyshev.chebfit(u_n, b_prev[i_3,:], deg = N_interpolation)
#         b_poly = np.polynomial.chebyshev.chebval(uu/(1-lam_new[i_3,0]*uu), poly_coeff)
#         b_new[i_3,:] = b_poly/(1-lam_new[i_3,0]*uu)**3
#     return b_new, uu, D_N_uu

def b_lambda_transformation_fixed_len_two_var(u_n, len_x, N_interpolation, b_prev, lam_new):
    b_new = np.zeros(b_prev.shape)
    min_lam = lam_new[:,0].min()
    if min_lam < 0:
            min_lam = 0.
    for i_3 in range(len_x):
        u_cut = u_n.max()/(1+min_lam*u_n.max()) #lambda is independant on u.
        D_N_uu, uu = cheb_newdom(u_n.size-1, 1/u_cut)
        poly_coeff = np.polynomial.chebyshev.chebfit(u_n, b_prev[i_3,:], deg = N_interpolation)
        b_poly = np.polynomial.chebyshev.chebval(uu/(1-lam_new[i_3,0]*uu), poly_coeff)
        b_new[i_3,:] = b_poly/(1-lam_new[i_3,0]*uu)**3
    return b_new, uu, D_N_uu

# Time propagation with RK4
from LIB_Horizon import fRK4


def RK4A(t_n, dt, known_func, N_beta_space, read_files):
    S = known_func
    K1 = fRK4(t_n, S, N_beta_space, read_files)
    K2 = fRK4(t_n + dt / 2, tuple(S[i] + (dt / 2) * K1[i] for i in range(len(S))), N_beta_space, read_files)
    K3 = fRK4(t_n + dt / 2, tuple(S[i] + (dt / 2) * K2[i] for i in range(len(S))), N_beta_space, read_files)
    K4 = fRK4(t_n + dt, tuple(S[i] + (dt) * K3[i] for i in range(len(S))), N_beta_space, read_files)
    Snew = tuple(S[i] + (dt / 6) * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]) for i in range(len(S)))
    return Snew


# def RK4B(t, interval_num, iteration_interval, known_func, N_beta_space, read_files):
#     N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
#     u_n = np.copy(u_n_reg)
#     u_n[-1] = 0.
#     S = known_func
#     #if interval_num == 0:
#     Super_S = tuple(np.array([S[i]]) for i in range(len(S)))
#     #else:
#     #    Super_S = S
#     N_t = t.size
#     n = interval_num * iteration_interval
#     if interval_num == 0:
#         indicator = 2
#     else:
#         indicator = 1
#     if n <= N_t-2:
#         while n <= iteration_interval + interval_num * iteration_interval - indicator:
#             if n <= N_t-2:
#                 dt = t[n+1] - t[n]
#                 S_n = tuple(Super_S[i][-1] for i in range(len(S)))
#                 Snp1 = RK4A(t[n],dt,S_n, N_beta_space, read_files)
#                 a_all, f_all, b_all, lam_all = Snp1
#                 # Low pass filter:
#                 a = low_pass_filter_with_u(a_all,beta_x3,u_n)
#                 f = low_pass_filter_with_u(f_all,beta_x3,u_n)
#                 b = low_pass_filter_with_u(b_all,beta_x3,u_n)
#                 lam = low_pass_filter_with_u(lam_all,beta_x3,u_n)
#                 Snp1filtered = a, f, b, lam
#                 if math.isinf(np.abs(Snp1filtered[0]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isnan(np.abs(Snp1filtered[0]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isinf(np.abs(Snp1filtered[1]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isnan(np.abs(Snp1filtered[1]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isinf(np.abs(Snp1filtered[2]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isnan(np.abs(Snp1filtered[2]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isinf(np.abs(Snp1filtered[3]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isnan(np.abs(Snp1filtered[3]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 Super_S = tuple(np.insert(Super_S[i] , Super_S[0][:,0,0].size , Snp1filtered[i], axis=0) for i in range(len(Super_S)))

#             else:
#                 print("break, you got to the end", n)
#                 break
#             n += 1
#         if interval_num != 0:
#             Super_S = tuple(np.delete(Super_S[i] , 0 , axis=0) for i in range(len(Super_S)))
#         return Super_S
#     print("Warning! you reached the limit")
#     return None

# def iterations_RK4(t, iteration_interval, known_func, N_beta_space, read_files):
#     interval_num = 0
#     N_t = t.size
#     Super_S = RK4B(t, interval_num, iteration_interval, known_func, N_beta_space, read_files)
#     Super_S_last_step = Super_S[0][-1, :, :], Super_S[1][-1, :, :], Super_S[2][-1, :, :], Super_S[3][-1, :, :]
#     interval_num += 1
#     while interval_num <= math.ceil(N_t / iteration_interval) - 1:
#         Super_S_interval = RK4B(t, interval_num, iteration_interval, Super_S_last_step, N_beta_space, read_files)
#         Super_S = tuple(
#             np.insert(Super_S[i], Super_S[0][:, 0, 0].size, Super_S_interval[i], axis=0) for i in range(len(Super_S)))
#         Super_S_last_step = Super_S_interval[0][-1, :, :], Super_S_interval[1][-1, :, :], Super_S_interval[2][-1, :, :], \
#         Super_S_interval[3][-1, :, :]
#         interval_num += 1
#     return Super_S


# def RK4B(t, interval_num, iteration_interval, known_func, N_beta_space, read_files):
#     N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
#     u_n = np.copy(u_n_reg)
#     u_n[-1] = 0.
#     S = known_func
#     #if interval_num == 0:
#     Super_S = tuple(np.array([S[i]]) for i in range(len(S)))
#     #else:
#     #    Super_S = S
#     N_t = t.size
#     n = interval_num * iteration_interval
#     print("n",n)
#     if interval_num == 0:
#         indicator = 2
#     else:
#         indicator = 1
#     if n <= N_t-2:
#         while n <= iteration_interval + interval_num * iteration_interval - indicator:
#             print("n in",n)
#             if n <= N_t-2:
#                 dt = t[n+1] - t[n]
#                 S_n = tuple(Super_S[i][-1] for i in range(len(S)))
#                 Snp1 = RK4A(t[n],dt,S_n, N_beta_space, read_files)
#                 a_all, f_all, b_all, lam_all = Snp1
#                 # Low pass filter:
#                 a = low_pass_filter_with_u(a_all,beta_x3,u_n)
#                 f = low_pass_filter_with_u(f_all,beta_x3,u_n)
#                 b = low_pass_filter_with_u(b_all,beta_x3,u_n)
#                 lam = low_pass_filter_with_u(lam_all,beta_x3,u_n)
#                 Snp1filtered = a, f, b, lam
#                 if math.isinf(np.abs(Snp1filtered[0]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isnan(np.abs(Snp1filtered[0]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isinf(np.abs(Snp1filtered[1]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isnan(np.abs(Snp1filtered[1]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isinf(np.abs(Snp1filtered[2]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isnan(np.abs(Snp1filtered[2]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isinf(np.abs(Snp1filtered[3]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 elif math.isnan(np.abs(Snp1filtered[3]).max()):
#                     print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                     break
#                 Super_S = tuple(np.insert(Super_S[i] , Super_S[0][:,0,0].size , Snp1filtered[i], axis=0) for i in range(len(Super_S)))

#             else:
#                 print("break", n)
#                 break
#             n += 1
#         if interval_num != 0:
#             Super_S = tuple(np.delete(Super_S[i] , 0 , axis=0) for i in range(len(Super_S)))
#             print(Super_S[0].shape)
#         return Super_S
#     print("Warning! you reached the limit")
#     return None

# def RK4B(t, n0, iteration_num, known_func, N_beta_space, read_files):
#     N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
#     u_n = np.copy(u_n_reg)
#     u_n[-1] = 0.

#     S = known_func
#     Super_S = tuple(np.array([S[i]]) for i in range(len(S)))
#     if n0 <= len(t)-1-1:
#         for n in range(n0, len(t)-1):
#             if n == n0 + iteration_num:
#                 return n, Super_S
#             dt = t[n+1] - t[n]
#             S_n = tuple(Super_S[i][-1] for i in range(len(S)))
#             Snp1 = RK4A(t[n],dt,S_n, N_beta_space, read_files)
#             a_all, f_all, b_all, lam_all = Snp1
#             # Low pass filter:
#             a = low_pass_filter_with_u(a_all,beta_x3,u_n)
#             f = low_pass_filter_with_u(f_all,beta_x3,u_n)
#             b = low_pass_filter_with_u(b_all,beta_x3,u_n)
#             lam = low_pass_filter_with_u(lam_all,beta_x3,u_n)
#             Snp1filtered = a, f, b, lam
#             if math.isinf(np.abs(Snp1filtered[0]).max()):
#                 print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                 break
#             elif math.isnan(np.abs(Snp1filtered[0]).max()):
#                 print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                 break
#             elif math.isinf(np.abs(Snp1filtered[1]).max()):
#                 print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                 break
#             elif math.isnan(np.abs(Snp1filtered[1]).max()):
#                 print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                 break
#             elif math.isinf(np.abs(Snp1filtered[2]).max()):
#                 print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                 break
#             elif math.isnan(np.abs(Snp1filtered[2]).max()):
#                 print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                 break
#             elif math.isinf(np.abs(Snp1filtered[3]).max()):
#                 print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                 break
#             elif math.isnan(np.abs(Snp1filtered[3]).max()):
#                 print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#                 break
#             Super_S = tuple(np.insert(Super_S[i] , Super_S[0][:,0,0].size , Snp1filtered[i], axis=0) for i in range(len(Super_S)))
#         return n, Super_S
#     return len(t), Super_S

# def RK4B(t,known_func, N_beta_space, read_files):
#     N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
#     u_n = np.copy(u_n_reg)
#     u_n[-1] = 0.

#     S = known_func
#     Super_S = tuple(np.array([S[i]]) for i in range(len(S)))
#     for n in range(len(t)-1):
#         dt = t[n+1] - t[n]
#         S_n = tuple(Super_S[i][-1] for i in range(len(S)))
#         Snp1 = RK4A(t[n],dt,S_n, N_beta_space, read_files)
#         a_all, f_all, b_all, lam_all = Snp1
#         # Low pass filter:
#         a = low_pass_filter_with_u(a_all,beta_x3,u_n)
#         f = low_pass_filter_with_u(f_all,beta_x3,u_n)
#         b = low_pass_filter_with_u(b_all,beta_x3,u_n)
#         lam = low_pass_filter_with_u(lam_all,beta_x3,u_n)
#         Snp1filtered = a, f, b, lam
#         if math.isinf(np.abs(Snp1filtered[0]).max()):
#             print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#             break
#         elif math.isnan(np.abs(Snp1filtered[0]).max()):
#             print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#             break
#         elif math.isinf(np.abs(Snp1filtered[1]).max()):
#             print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#             break
#         elif math.isnan(np.abs(Snp1filtered[1]).max()):
#             print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#             break
#         elif math.isinf(np.abs(Snp1filtered[2]).max()):
#             print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#             break
#         elif math.isnan(np.abs(Snp1filtered[2]).max()):
#             print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#             break
#         elif math.isinf(np.abs(Snp1filtered[3]).max()):
#             print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#             break
#         elif math.isnan(np.abs(Snp1filtered[3]).max()):
#             print(f"Divergent occurred at step - {n+1}, which is t(n+1)={t[n+1]}")
#             break
#         Super_S = tuple(np.insert(Super_S[i] , Super_S[0][:,0,0].size , Snp1filtered[i], axis=0) for i in range(len(Super_S)))
#     return Super_S

# def RK4B(t, known_func, N_beta_space, read_files):
#     num_save = [1000* (i+1) for i in range(len(t)//1000)]
#     N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
#     u_n = np.copy(u_n_reg)
#     u_n[-1] = 0.
#
#     S = known_func
#     Super_S = tuple(np.array([S[i]]) for i in range(len(S)))
#     for n in range(len(t) - 1):
#         dt = t[n + 1] - t[n]
#         S_n = tuple(Super_S[i][-1] for i in range(len(S)))
#         Snp1 = RK4A(t[n], dt, S_n, N_beta_space, read_files)
#         a_all, f_all, b_all, lam_all = Snp1
#         # Low pass filter:
#         a = low_pass_filter_with_u(a_all, beta_x3, u_n)
#         f = low_pass_filter_with_u(f_all, beta_x3, u_n)
#         b = low_pass_filter_with_u(b_all, beta_x3, u_n)
#         lam = low_pass_filter_with_u(lam_all, beta_x3, u_n)
#         Snp1filtered = a, f, b, lam
#         if math.isinf(np.abs(Snp1filtered[0]).max()):
#             print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
#             break
#         elif math.isnan(np.abs(Snp1filtered[0]).max()):
#             print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
#             break
#         elif math.isinf(np.abs(Snp1filtered[1]).max()):
#             print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
#             break
#         elif math.isnan(np.abs(Snp1filtered[1]).max()):
#             print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
#             break
#         elif math.isinf(np.abs(Snp1filtered[2]).max()):
#             print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
#             break
#         elif math.isnan(np.abs(Snp1filtered[2]).max()):
#             print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
#             break
#         elif math.isinf(np.abs(Snp1filtered[3]).max()):
#             print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
#             break
#         elif math.isnan(np.abs(Snp1filtered[3]).max()):
#             print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
#             break
#         Super_S = tuple(
#             np.insert(Super_S[i], Super_S[0][:, 0, 0].size, Snp1filtered[i], axis=0) for i in range(len(Super_S)))
#         if n in num_save:
#             #np.savez(f'Super_S_{n}.npz', *Super_S)
#             print("Last saved step is n=", n)
#     return Super_S

# Cheating: You shell go back to the last RK4B (one above)
def RK4B(t, known_func, N_beta_space, read_files):
    print(known_func[2].shape)
    num_save = [1000* (i+1) for i in range(len(t)//1000)]
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    u_n = np.copy(u_n_reg)
    u_n[-1] = 0.

    S = known_func
    Super_S = tuple(np.array([S[i]]) for i in range(len(S)))
    print(Super_S[2].shape)
    for n in range(len(t) - 1):
        dt = t[n + 1] - t[n]
        S_n = tuple(Super_S[i][-1] for i in range(len(S)))
        Snp1 = RK4A(t[n], dt, S_n, N_beta_space, read_files)
        a_all, f_all, b_all, lam_all = Snp1
        # Low pass filter:
        a = low_pass_filter_with_u(a_all, beta_x3, u_n)
        f = low_pass_filter_with_u(f_all, beta_x3, u_n)
        b = low_pass_filter_with_u(b_all, beta_x3, u_n)
        lam = low_pass_filter_with_u(lam_all, beta_x3, u_n)   # Filtering a so need also lam
        Snp1filtered = a, f, b, lam
        if math.isinf(np.abs(Snp1filtered[0]).max()):
            print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
            break
        elif math.isnan(np.abs(Snp1filtered[0]).max()):
            print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
            break
        elif math.isinf(np.abs(Snp1filtered[1]).max()):
            print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
            break
        elif math.isnan(np.abs(Snp1filtered[1]).max()):
            print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
            break
        elif math.isinf(np.abs(Snp1filtered[2]).max()):
            print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
            break
        elif math.isnan(np.abs(Snp1filtered[2]).max()):
            print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
            break
        elif math.isinf(np.abs(Snp1filtered[3]).max()):
            print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
            break
        elif math.isnan(np.abs(Snp1filtered[3]).max()):
            print(f"Divergent occurred at step - {n + 1}, which is t(n+1)={t[n + 1]}")
            break
        Super_S = tuple(
            np.insert(Super_S[i], Super_S[0][:, 0, 0].size, Snp1filtered[i], axis=0) for i in range(len(Super_S)))
        if n in num_save:
            #np.savez(f'Super_S_{n}.npz', *Super_S)
            print("Last saved step is n=", n)
    return Super_S

# def Iterations_RK4(time_data, N_beta_space, path, read_files):
#     t_i, t_f, dt, slice_range = time_data
#     t_prev = np.arange(0, t_i, dt)
#     t_new = np.arange(t_i - dt, t_f, dt)
#     t_new_true = np.delete(t_new, 0, axis=0)
#     t = np.concatenate((t_prev, t_new_true))
#
#     print(math.ceil(t_i))
#     data = np.load(f'{path}known_func_t_saved_{math.ceil(t_i)}_sliced.npz')
#     known_func_t_load_old = tuple(data[key] for key in data.files)
#     known_func = tuple(known_func_t_load_old[i][-1, :, :] for i in range(len(known_func_t_load_old)))
#
#     # Change lambda here.
#     a, f, b, lam = known_func
#     lam_fixed = -1. + (-2 * a) ** (1 / 4)
#     known_func_fixed = a, f, b, lam_fixed
#
#     known_func_t_next_interval = RK4B(t_new, known_func_fixed, N_beta_space, read_files)
#     known_func_t_next_interval = tuple(
#         np.delete(known_func_t_next_interval[i], 0, axis=0) for i in range(len(known_func_t_next_interval)))
#
#     # Slicing
#     known_func_t_load_cut = tuple(
#         known_func_t_next_interval[i][::slice_range] for i in range(len(known_func_t_next_interval)))
#     if known_func_t_load_cut[0][-1, 0, 0] != known_func_t_next_interval[0][-1, 0, 0]:
#         known_func_t_load_cut = tuple(
#             np.append(known_func_t_load_cut[i], np.array([known_func_t_next_interval[i][-1]]), axis=0) for i in
#             range(len(known_func_t_next_interval)))
#
#     np.savez(f'{path}known_func_t_saved_{math.ceil(t_f)}_sliced.npz', *known_func_t_load_cut)
#     return known_func_t_load_cut

def Iterations_RK4(time_data, N_beta_space, known_func_t_0, path, read_files):
    t_i, t_f, dt, slice_range = time_data
    N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
    u_n = np.copy(u_n_reg)
    u_n[-1] = 0.
    #U, X3 = np.meshgrid(u_n, x3_n)
    if int(t_i) != 0:
        t_prev = np.arange(0, t_i, dt)
        t_new = np.arange(t_i - dt, t_f, dt)
        #t_new_true = np.delete(t_new, 0, axis=0)
        #t = np.concatenate((t_prev, t_new_true))

        print(math.ceil(t_i))
        data = np.load(f'{path}known_func_t_saved_{math.ceil(t_i)}_sliced.npz')
        known_func_t_load_old = tuple(data[key] for key in data.files)
        known_func = tuple(known_func_t_load_old[i][-1, :, :] for i in range(len(known_func_t_load_old)))

        # Change lambda here.
        a, f, b, lam = known_func
        #known_func_fixed = known_func
        # lam_fixed = lam #-1. + (-2 * a) ** (1 / 4) # for later (Just want to finish the previous run)
        # N_interpolation = N_u//2
        # len_x3 = x3_n.size
        # #b_fixed, u_n_fixed, D_N_u_fixed = b_lambda_transformation_fixed_len_two_var(u_n, len_x3, N_interpolation, b, lam_fixed)
        #
        # u_n_reg_fixed = np.copy(u_n_fixed)
        # u_n_reg_fixed[-1] = 1.
        # beta_u_fixed = 1/u_n_fixed.max()
        #
        # N_beta_space_fixed = N_u, N_x3, beta_u_fixed, beta_x3, D_N_u_fixed, u_n_reg_fixed, x3_n
        #
        # known_func_fixed = a, f, b, lam
        # known_func_fixed = known_func

        known_func_t_next_interval = RK4B(t_new, known_func, N_beta_space, read_files)
        known_func_t_next_interval = tuple(
            np.delete(known_func_t_next_interval[i], 0, axis=0) for i in range(len(known_func_t_next_interval)))

        # Slicing
        known_func_t_load_cut = tuple(
            known_func_t_next_interval[i][::slice_range] for i in range(len(known_func_t_next_interval)))
        # if known_func_t_load_cut[0][-1, 0, 0] != known_func_t_next_interval[0][-1, 0, 0]:
        #     known_func_t_load_cut = tuple(
        #         np.append(known_func_t_load_cut[i], np.array([known_func_t_next_interval[i][-1]]), axis=0) for i in
        #         range(len(known_func_t_next_interval)))

        np.savez(f'{path}known_func_t_saved_{math.ceil(t_f)}_sliced.npz', *known_func_t_load_cut)
    else:
        t_new = np.arange(t_i, t_f, dt)
        known_func = known_func_t_0

        # Change lambda here.
        #a, f, b, lam = known_func  # for later (Just want to finish the previous run)
        # lam_fixed = -1. + (-2 * a) ** (1 / 4)
        #
        # N_interpolation = N_u - 5
        # len_x3 = x3_n.size
        # b_fixed, u_n_fixed, D_N_u_fixed = b_lambda_transformation_fixed_len_two_var(u_n, len_x3, N_interpolation, b, lam_fixed)
        #
        # u_n_reg_fixed = np.copy(u_n_fixed)
        # u_n_reg_fixed[-1] = 1.
        # beta_u_fixed = 1/u_n_fixed.max()

        # N_beta_space_fixed = N_u, N_x3, beta_u_fixed, beta_x3, D_N_u_fixed, u_n_reg_fixed, x3_n
        #
        # known_func_fixed = a, f, b_fixed, lam_fixed
        # known_func_fixed = known_func
        # N_beta_space_fixed = N_beta_space

        known_func_t_next_interval = RK4B(t_new, known_func, N_beta_space, read_files)

        # Slicing
        known_func_t_load_cut = tuple(
            known_func_t_next_interval[i][::slice_range] for i in range(len(known_func_t_next_interval)))
        # if known_func_t_load_cut[0][-1, 0, 0] != known_func_t_next_interval[0][-1, 0, 0]:
        #     known_func_t_load_cut = tuple(
        #         np.append(known_func_t_load_cut[i], np.array([known_func_t_next_interval[i][-1]]), axis=0) for i in
        #         range(len(known_func_t_next_interval)))

        np.savez(f'{path}known_func_t_saved_{math.ceil(t_f)}_sliced.npz', *known_func_t_load_cut)
    return known_func_t_load_cut

# In case I messed up with the new code:
# def Iterations_RK4(time_data, N_beta_space, known_func_t_0, path, read_files):
#     t_i, t_f, dt, slice_range = time_data
#     N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n = N_beta_space
#     u_n = np.copy(u_n_reg)
#     u_n[-1] = 0.
#     U, X3 = np.meshgrid(u_n, x3_n)
#     if int(t_i) != 0:
#         t_prev = np.arange(0, t_i, dt)
#         t_new = np.arange(t_i - dt, t_f, dt)
#         t_new_true = np.delete(t_new, 0, axis=0)
#         t = np.concatenate((t_prev, t_new_true))
#
#         print(math.ceil(t_i))
#         data = np.load(f'{path}known_func_t_saved_{math.ceil(t_i)}_sliced.npz')
#         known_func_t_load_old = tuple(data[key] for key in data.files)
#         known_func = tuple(known_func_t_load_old[i][-1, :, :] for i in range(len(known_func_t_load_old)))
#
#         # Change lambda here.
#         a, f, b, lam = known_func
#         lam_fixed = -1. + (-2 * a) ** (1 / 4) # for later (Just want to finish the previous run)
#         delta_lam = lam_fixed - lam
#         b_fixed = (1+delta_lam * U)**3 * b # Fixing b as a result of fixing lambda
#         known_func_fixed = a, f, b_fixed, lam_fixed
#         # known_func_fixed = known_func
#
#         known_func_t_next_interval = RK4B(t_new, known_func_fixed, N_beta_space, read_files)
#         known_func_t_next_interval = tuple(
#             np.delete(known_func_t_next_interval[i], 0, axis=0) for i in range(len(known_func_t_next_interval)))
#
#         # Slicing
#         known_func_t_load_cut = tuple(
#             known_func_t_next_interval[i][::slice_range] for i in range(len(known_func_t_next_interval)))
#         # if known_func_t_load_cut[0][-1, 0, 0] != known_func_t_next_interval[0][-1, 0, 0]:
#         #     known_func_t_load_cut = tuple(
#         #         np.append(known_func_t_load_cut[i], np.array([known_func_t_next_interval[i][-1]]), axis=0) for i in
#         #         range(len(known_func_t_next_interval)))
#
#         np.savez(f'{path}known_func_t_saved_{math.ceil(t_f)}_sliced.npz', *known_func_t_load_cut)
#     else:
#         t_new = np.arange(t_i, t_f, dt)
#         known_func = known_func_t_0
#
#         # Change lambda here.
#         a, f, b, lam = known_func  # for later (Just want to finish the previous run)
#         lam_fixed = -1. + (-2 * a) ** (1 / 4)
#         delta_lam = lam_fixed - lam
#         b_fixed = (1 + delta_lam * U) ** 3 * b  # Fixing b as a result of fixing lambda
#         known_func_fixed = a, f, b_fixed, lam_fixed
#         # known_func_fixed = known_func
#
#         known_func_t_next_interval = RK4B(t_new, known_func_fixed, N_beta_space, read_files)
#
#         # Slicing
#         known_func_t_load_cut = tuple(
#             known_func_t_next_interval[i][::slice_range] for i in range(len(known_func_t_next_interval)))
#         # if known_func_t_load_cut[0][-1, 0, 0] != known_func_t_next_interval[0][-1, 0, 0]:
#         #     known_func_t_load_cut = tuple(
#         #         np.append(known_func_t_load_cut[i], np.array([known_func_t_next_interval[i][-1]]), axis=0) for i in
#         #         range(len(known_func_t_next_interval)))
#
#         np.savez(f'{path}known_func_t_saved_{math.ceil(t_f)}_sliced.npz', *known_func_t_load_cut)
#     return known_func_t_load_cut