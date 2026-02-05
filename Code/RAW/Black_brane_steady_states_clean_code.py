# Import
import math
import sys

# from scipy.integrate import cumulative_trapezoid
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy as sp
import sympy as smp
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint, quad
from sympy.interactive import printing

printing.init_printing(use_latex=True)
import winsound

from sympy.parsing.mathematica import parse_mathematica

from LIB_Horizon import (
    fRK4,
    kretschmann_scalar,
    readfiledouble,
    readfilesingle,
    solve_y1_y2,
    solve_y3_y4,
    solve_y5,
    solve_y6,
    solve_y7_y8,
    solve_y9,
)

# My modules
from My_PDE_Horizon import (
    RK4A,
    RK4B,
    Iterations_RK4,
    _alt,
    alt,
    arrconsec,
    b_lambda_transformation_fixed_len_two_var,
    cesnoc,
    cheb,
    cheb_newdom,
    consec,
    fourier_derivative,
    fourier_derivative_two_variables,
    fourier_dom,
    low_pass_filter,
    low_pass_filter_with_u,
)

# %%
alpha_temp = 1.0
N_u, beta_u = 2**5 - 1, 0.9
N_x3, beta_x3 = 2**8, 1000.0  # /100.
D_N_u, u_n = cheb_newdom(N_u, beta_u)
x3_n = fourier_dom(N_x3, beta_x3)
U, X3 = np.meshgrid(u_n, x3_n)
l = beta_x3 / 4
slope = 0.006  # *75.
beta = 10.0

# Setting the Riemann problem boundary conditions
P_L, P_R = 0.5, 0.25
vz_L, vz_R = 0.0, -0.25

# Given the pressure and the velocity, we conclude that
a_L, a_R = -(2 / 3) * ((3 + vz_L**2) / (1 - vz_L**2)) * P_L, -(2 / 3) * ((3 + vz_R**2) / (1 - vz_R**2)) * P_R
f_L, f_R = -4 * (vz_L / (1 - vz_L**2)) * P_L, -4 * (vz_R / (1 - vz_R**2)) * P_R
b4_L, b4_R = -(4 / 3) * (vz_L**2 / (1 - vz_L**2)) * P_L, -(4 / 3) * (vz_R**2 / (1 - vz_R**2)) * P_R
print("a_L=", a_L)
print("a_R=", a_R)
a_lambda = lambda u, x3: (a_L + a_R) / 2 + ((a_L - a_R) / 2) * (
    (np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope)))
) / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
f_lambda = lambda u, x3: (f_L + f_R) / 2 + ((f_L - f_R) / 2) * (
    (np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope)))
) / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
b_lambda = lambda u, x3: u * (
    (b4_L + b4_R) / 2
    + ((b4_L - b4_R) / 2)
    * ((np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope))))
    / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
)

b1 = b_lambda(U, X3)
f1 = f_lambda(U, X3)
a1 = a_lambda(U, X3)

l1 = 0 * ((-2 * a1) ** (1 / 4) - 1.0)

par = beta_x3, N_u
# f_test = (1/2) * fourier_derivative_two_variables(a1,par)
dp = np.abs((P_L - P_R) / (P_L + P_R))
P_R_true = -(3 / 2) * ((1 - vz_R**2) / (3 + vz_R**2)) * a1[0, 0]
P_L_true = -a1[len(x3_n) // 2, 0] / 2
dp_a = np.abs((P_L_true - P_R_true) / (P_L_true + P_R_true))
print("The true P_R=", P_R_true)
print("The true P_L=", P_L_true)

print("a(x_L)=", a1.max())
print("a(x_R)=", a1.min())
print("f(x_L)=", f1.min())
print("f(x_R)=", f1.max())
print("max(|a|)=", 1 / (2 * beta_u))
print("dp=", dp)
print("dpa=", dp_a)

plt.figure("initial profile a")
plt.clf()
plt.plot(x3_n, a1[:, 0], "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
plt.axhline(y=-0.5 * alpha_temp, color="r", label="Non perturbed BH radius", ls="--")
plt.grid(True)
plt.ylabel("$a_4(x_3)$")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

plt.figure("initial profile f")
plt.clf()
plt.plot(x3_n, f1[:, 0], "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
# plt.axhline(y = -0.5, color = 'r', label = 'Non perturbed BH radius', ls = '--')
plt.grid(True)
plt.ylabel("df")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

print("minimal r_H=", min(-l1[:, 0] + (-2 * a1[:, 0]) ** (1 / 4)))
print("(no lambda) minimal r_H=", min((-2 * a1[:, 0]) ** (1 / 4)))

# from this graph you know how to choose beta_u
plt.figure("Radial Horizon without lambda")
plt.clf()
plt.title("Radial Horizon without $\\lambda$")
plt.plot(x3_n, (-2 * a1[:, 0]) ** (1 / 4), "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
plt.axhline(y=np.mean((-2 * a1[:, 0]) ** (1 / 4)), color="r", label="Non perturbed BH radius", ls="--")
plt.grid(True)
plt.ylabel("$r_H(x_3)$")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

plt.figure("Radial Horizon")
plt.clf()
plt.title("Radial Horizon with $\\lambda$")
plt.plot(x3_n, -l1[:, 0] + (-2 * a1[:, 0]) ** (1 / 4), "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
plt.axhline(y=np.mean(-l1[:, 0] + (-2 * a1[:, 0]) ** (1 / 4)), color="r", label="Non perturbed BH radius", ls="--")
plt.grid(True)
plt.ylabel("$r_H(x_3)$")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

plt.figure("Radial Horizon in u without lambda")
plt.clf()
plt.title("Radial Horizon $u_H$ without $\\lambda$ (Approximatation only)")
plt.plot(x3_n, 1 / (-2 * a1[:, 0]) ** (1 / 4), "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
# plt.axhline(y = np.mean(1/(-2*a1[:,0])**(1/4)), color = 'r', label = 'Non perturbed BH radius', ls = '--')
plt.grid(True)
plt.ylabel("$u_H(x_3)$")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

# %%
# New Approach:
interval_saver = 50.0
Steps = 6

interval_saver = 100.0
Steps = 3

t_i = np.array([interval_saver * i for i in range(0, Steps)])
t_f = np.array([interval_saver * (i + 1) for i in range(0, Steps)])
dt, slice_range = 0.01, 100
if int(interval_saver / dt) % slice_range == 0:
    print("You can go")
else:
    raise RuntimeError("I have some problems")

alpha_temp = 1.0
N_u, beta_u = 2**5 - 1, 0.9
N_x3, beta_x3 = 2**8, 1000.0  # /100.
D_N_u, u_n = cheb_newdom(N_u, beta_u)
x3_n = fourier_dom(N_x3, beta_x3)
u_n_reg = np.copy(u_n)
u_n_reg[-1] = 1.0
U, X3 = np.meshgrid(u_n, x3_n)

N_beta_space = N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n
l = beta_x3 / 4
slope = 0.006  # *75.
beta = 10.0

# Setting the Riemanns problem boundary conditions
P_L, P_R = 0.5, 0.25
vz_L, vz_R = 0.0, -0.25

# Given the pressure and the velocity, we conclude that
a_L, a_R = -(2 / 3) * ((3 + vz_L**2) / (1 - vz_L**2)) * P_L, -(2 / 3) * ((3 + vz_R**2) / (1 - vz_R**2)) * P_R
f_L, f_R = -4 * (vz_L / (1 - vz_L**2)) * P_L, -4 * (vz_R / (1 - vz_R**2)) * P_R
b4_L, b4_R = -(4 / 3) * (vz_L**2 / (1 - vz_L**2)) * P_L, -(4 / 3) * (vz_R**2 / (1 - vz_R**2)) * P_R

a_lambda = lambda u, x3: (a_L + a_R) / 2 + ((a_L - a_R) / 2) * (
    (np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope)))
) / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
f_lambda = lambda u, x3: (f_L + f_R) / 2 + ((f_L - f_R) / 2) * (
    (np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope)))
) / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
b_lambda = lambda u, x3: u * (
    (b4_L + b4_R) / 2
    + ((b4_L - b4_R) / 2)
    * ((np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope))))
    / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
)

b1 = b_lambda(U, X3)
f31 = f_lambda(U, X3)
a1 = a_lambda(U, X3)
lam1 = 0 * (-1.0 + (-2 * a1) ** (1 / 4))

known_func_t_0 = a1, f31, b1, lam1

A_21_y1y2, A_22_y1y2, g_2_y1y2 = readfiledouble("y1y2")
A_21_y3y4, A_22_y3y4, g_2_y3y4 = readfiledouble("y3y4")
A_y_5, g_y5 = readfilesingle("y5")
A_y_6, g_y6 = readfilesingle("y6")
A_21_y7y8, A_22_y7y8, g_2_y7y8 = readfiledouble("y7y8")
read_files = (
    A_21_y1y2,
    A_22_y1y2,
    g_2_y1y2,
    A_21_y3y4,
    A_22_y3y4,
    g_2_y3y4,
    A_y_5,
    g_y5,
    A_y_6,
    g_y6,
    A_21_y7y8,
    A_22_y7y8,
    g_2_y7y8,
)

path = r"C:\Users\matan\PycharmProjects\Black_brane_steady_state_stabilized_Horizon_Fixed_lambda/Saved data/Saved iteration L 1000 slope 0_006 P_L 0_5 P_R 0_25 vL 0 vR m0_25 no_lambda/"
path = r"saved_data\Saved iteration L 1000 slope 0_006 P_L 0_5 P_R 0_25 vL 0 vR m0_25 no_lambda"

start_time = time.time()
for i in range(len(t_i)):
    time_data = t_i[i], t_f[i], dt, slice_range
    print(f"Current iteration: ({t_i[i], t_f[i]})")
    known_func_t_next_interval = Iterations_RK4(time_data, N_beta_space, known_func_t_0, path, read_files)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
print("--- %s hours ---" % ((time.time() - start_time) / 3600))
winsound.Beep(640, 1500)  # Beep at 1000 Hz for 1 second
winsound.Beep(440, 1500)  # Beep at 1000 Hz for 1 second
winsound.Beep(640, 1500)  # Beep at 1000 Hz for 1 second

# %%

alpha_temp = 2.0
N_u, beta_u = 2**5 - 1, 0.95
N_x3, beta_x3 = 2**8, 1000.0
D_N_u, u_n = cheb_newdom(N_u, beta_u)
x3_n = fourier_dom(N_x3, beta_x3)
U, X3 = np.meshgrid(u_n, x3_n)
l = beta_x3 / 4
slope = 30.0
A_0 = 1.0

alpha = 1.082
beta = 1 / 2
a_lambda = (
    lambda u, x3: -A_0
    * (
        (1 + alpha * np.tanh(beta * np.tanh((x3 + l) / slope)))
        * (1 + alpha * np.tanh(beta * np.tanh((-x3 + l) / slope)))
    )
    / ((1 + alpha * np.tanh(beta * np.tanh((l) / slope))) * (1 + alpha * np.tanh(beta * np.tanh((l) / slope))))
)
# l_lambda = lambda u,x3: 0*x3
a1_non_normalized = a_lambda(U, X3)
# l1 = l_lambda(U,X3)
a1 = -0.5 * alpha_temp - (a1_non_normalized.max() + a1_non_normalized.min()) / 2 + a1_non_normalized
l1 = -1.1 + (-2 * a1) ** (1 / 4)

par = beta_x3, N_u
f_test = (1 / 2) * fourier_derivative_two_variables(a1, par)
dp = np.abs((a1.max() - a1.min()) / (a1.max() + a1.min()))

print("a(x_L)=", a1.max())
print("a(x_R)=", a1.min())
print("max(|a|)=", 1 / (2 * beta_u))
print("dp=", dp)
# print("max(J/P0)=",(2/3)*np.sqrt(5*dp**2+np.sqrt(1-dp**2)-1))

plt.figure("initial profile a")
plt.clf()
plt.plot(x3_n, a1[:, 0], "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
plt.axhline(y=-0.5 * alpha_temp, color="r", label="Non perturbed BH radius", ls="--")
plt.grid(True)
plt.ylabel("$a_4(x_3)$")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

plt.figure("initial profile f")
plt.clf()
plt.plot(x3_n, f_test[:, 0], "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
# plt.axhline(y = -0.5, color = 'r', label = 'Non perturbed BH radius', ls = '--')
plt.grid(True)
plt.ylabel("df")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

print("minimal r_H=", min(-l1[:, 0] + (-2 * a1[:, 0]) ** (1 / 4)))
print("(no lambda) minimal r_H=", min((-2 * a1[:, 0]) ** (1 / 4)))

# from this graph you know how to choose beta_u
plt.figure("Radial Horizon without lambda")
plt.clf()
plt.title("Radial Horizon without $\\lambda$")
plt.plot(x3_n, (-2 * a1[:, 0]) ** (1 / 4), "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
plt.axhline(y=np.mean((-2 * a1[:, 0]) ** (1 / 4)), color="r", label="Non perturbed BH radius", ls="--")
plt.grid(True)
plt.ylabel("$r_H(x_3)$")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

plt.figure("Radial Horizon")
plt.clf()
plt.title("Radial Horizon with $\\lambda$")
plt.plot(x3_n, -l1[:, 0] + (-2 * a1[:, 0]) ** (1 / 4), "b.")
plt.axvline(x=0, color="g", label="z=0", ls="--")
plt.axhline(y=np.mean(-l1[:, 0] + (-2 * a1[:, 0]) ** (1 / 4)), color="r", label="Non perturbed BH radius", ls="--")
plt.grid(True)
plt.ylabel("$r_H(x_3)$")
plt.xlabel("$x_3$")
plt.legend()
plt.show()

# %%

# First time chunk propagation:

t = np.arange(0.0, 50.0, 0.01)
slice_range = 100
alpha_temp = 1.0
N_u, beta_u = 2**5 - 1, 0.95
N_x3, beta_x3 = 2**8, 1000.0
D_N_u, u_n = cheb_newdom(N_u, beta_u)
x3_n = fourier_dom(N_x3, beta_x3)
u_n_reg = np.copy(u_n)
u_n_reg[-1] = 1.0
U, X3 = np.meshgrid(u_n, x3_n)

N_beta_space = N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n
l = beta_x3 / 4
slope = 0.006
beta = 10.0

# Setting the Riemanns problem boundary conditions
P_L, P_R = 0.5, 0.25
vz_L, vz_R = 0.0, 0.0

# Given the pressure and the velocity, we conclude that
a_L, a_R = -(2 / 3) * ((3 + vz_L**2) / (1 - vz_L**2)) * P_L, -(2 / 3) * ((3 + vz_R**2) / (1 - vz_R**2)) * P_R
f_L, f_R = -4 * (vz_L / (1 - vz_L**2)) * P_L, -4 * (vz_R / (1 - vz_R**2)) * P_R
b4_L, b4_R = -(4 / 3) * (vz_L**2 / (1 - vz_L**2)) * P_L, -(4 / 3) * (vz_R**2 / (1 - vz_R**2)) * P_R

a_lambda = lambda u, x3: (a_L + a_R) / 2 - ((a_L - a_R) / 2) * (
    (np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope)))
)
f_lambda = lambda u, x3: (f_L + f_R) / 2 - ((f_L - f_R) / 2) * (
    (np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope)))
)
b_lambda = lambda u, x3: u * (
    (b4_L + b4_R) / 2
    - ((b4_L - b4_R) / 2) * ((np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope))))
)

b1 = b_lambda(U, X3)
f1 = f_lambda(U, X3)
a1 = a_lambda(U, X3)

b1 = b_lambda(U, X3)
f31 = f_lambda(U, X3)
a1 = a_lambda(U, X3)
# l1 = (-2*a1)**(1/4)-1.
lam1 = -1.0 + (-2 * a1) ** (1 / 4)

known_func = a1, f31, b1, lam1

A_21_y1y2, A_22_y1y2, g_2_y1y2 = readfiledouble("y1y2")
A_21_y3y4, A_22_y3y4, g_2_y3y4 = readfiledouble("y3y4")
A_y_5, g_y5 = readfilesingle("y5")
A_y_6, g_y6 = readfilesingle("y6")
A_21_y7y8, A_22_y7y8, g_2_y7y8 = readfiledouble("y7y8")
read_files = (
    A_21_y1y2,
    A_22_y1y2,
    g_2_y1y2,
    A_21_y3y4,
    A_22_y3y4,
    g_2_y3y4,
    A_y_5,
    g_y5,
    A_y_6,
    g_y6,
    A_21_y7y8,
    A_22_y7y8,
    g_2_y7y8,
)

path = r"C:\Users\matan\PycharmProjects\Black_brane_steady_state_stabilized_Horizon/Saved iteration L 1000 slope 0_006 P_L 0_5 P_R 0_25 vL 0 vR 0/"
path = r"saved_data\Saved iteration L 1000 slope 0_006 P_L 0_5 P_R 0_25 vL 0 vR 0"

start_time = time.time()
known_func_t = RK4B(t, known_func, N_beta_space, read_files)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
print("--- %s hours ---" % ((time.time() - start_time) / 3600))
winsound.Beep(440, 1500)  # Beep at 1000 Hz for 1 second
winsound.Beep(540, 1500)  # Beep at 1000 Hz for 1 second
winsound.Beep(640, 1500)  # Beep at 1000 Hz for 1 second

# Slicing
print(known_func_t[0].shape)
slice_range = 100
known_func_t_load_cut = tuple(known_func_t[i][::slice_range] for i in range(len(known_func_t)))
if known_func_t_load_cut[0][-1, 0, 0] != known_func_t[0][-1, 0, 0]:
    known_func_t_load_cut = tuple(
        np.append(known_func_t_load_cut[i], np.array([known_func_t[i][-1]]), axis=0) for i in range(len(known_func_t))
    )

    np.savez(f"{path}known_func_t_saved_{math.ceil(t[-1])}_sliced.npz", *known_func_t_load_cut)
# %%
print(np.array([50.0 * (i + 1) for i in range(0, 8)]))
# %%
t_i = np.array([50.0 * (i + 1) for i in range(0, 8)])
t_f = np.array([50.0 * (i + 2) for i in range(0, 8)])
dt, slice_range = 0.01, 100

# alpha_temp = 2.
# N_u, beta_u = 2**5-1, 0.98
# N_x3, beta_x3 = 2**8, 15000.
# D_N_u, u_n = cheb_newdom(N_u,beta_u)
# x3_n = fourier_dom(N_x3,beta_x3)
# u_n_reg = np.copy(u_n)
# u_n_reg[-1] = 1.
#
# N_beta_space = N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n
#
# A_21_y1y2, A_22_y1y2, g_2_y1y2 = readfiledouble('y1y2')
# A_21_y3y4, A_22_y3y4, g_2_y3y4 = readfiledouble('y3y4')
# A_y_5, g_y5 = readfilesingle('y5')
# A_y_6, g_y6 = readfilesingle('y6')
# A_21_y7y8, A_22_y7y8, g_2_y7y8 = readfiledouble('y7y8')
# read_files = A_21_y1y2, A_22_y1y2, g_2_y1y2, A_21_y3y4, A_22_y3y4, g_2_y3y4, A_y_5, g_y5, A_y_6, g_y6, A_21_y7y8, A_22_y7y8, g_2_y7y8
#
# path = r'C:\Users\matan\PycharmProjects\Black_brane_steady_state_stabilized_Horizon/Saved iteration L 15000 slope 450 A 2_8/'


start_time = time.time()
for i in range(len(t_i)):
    time_data = t_i[i], t_f[i], dt, slice_range
    print(f"Current iteration: ({t_i[i], t_f[i]})")
    known_func_t_next_interval = Iterations_RK4(time_data, N_beta_space, known_func_t0, path, read_files)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
print("--- %s hours ---" % ((time.time() - start_time) / 3600))
winsound.Beep(640, 1500)  # Beep at 1000 Hz for 1 second
winsound.Beep(440, 1500)  # Beep at 1000 Hz for 1 second
winsound.Beep(640, 1500)  # Beep at 1000 Hz for 1 second

# %%
# print(np.array([50.*(i+1) for i in range(19,40)]))
# print(np.array([50.*(i+2) for i in range(19,40)]))
# path = r'C:\Users\matan\PycharmProjects\Black_brane_steady_state_stabilized_Horizon/Saved iteration L 15000 slope 450 A 2_8/'
# Directory_check = np.array([0,1,2])
# np.savez(f'{path}Directory_check.npz', *Directory_check)
# Setting known function t_0:


alpha_temp = 2.0
N_u, beta_u = 2**5 - 1, 0.98
N_x3, beta_x3 = 2**8, 15000.0
D_N_u, u_n = cheb_newdom(N_u, beta_u)
x3_n = fourier_dom(N_x3, beta_x3)
u_n_reg = np.copy(u_n)
u_n_reg[-1] = 1.0
U, X3 = np.meshgrid(u_n, x3_n)

N_beta_space = N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n
l = beta_x3 / 4
slope = 450.0
A_0 = 2.8
alpha = 1.082
beta = 1 / 2
a_lambda = (
    lambda u, x3: -A_0
    * (
        (1 + alpha * np.tanh(beta * np.tanh((x3 + l) / slope)))
        * (1 + alpha * np.tanh(beta * np.tanh((-x3 + l) / slope)))
    )
    / ((1 + alpha * np.tanh(beta * np.tanh((l) / slope))) * (1 + alpha * np.tanh(beta * np.tanh((l) / slope))))
)
f3_lambda = lambda u, x3: 0 * x3
# l_lambda = lambda u,x3: 0*x3
b_lambda = lambda u, x3: 0 * x3

b1 = b_lambda(U, X3)
# lam1 = l_lambda(U,X3)
f31 = f3_lambda(U, X3)
a1_non_normalized = a_lambda(U, X3)
a1 = -0.5 * alpha_temp - (a1_non_normalized.max() + a1_non_normalized.min()) / 2 + a1_non_normalized
lam1 = -1.0 + (-2 * a1) ** (1 / 4)
# par = beta_x3, N_u
# f31 = 10*(1/2) * fourier_derivative_two_variables(a1,par)
# a1new = -0.5 + (a1+0.5)/100
known_func_t0 = a1, f31, b1, lam1
# %%
t_i = np.array([50.0 * (i + 1) for i in range(19, 40)])
t_f = np.array([50.0 * (i + 2) for i in range(19, 40)])
dt, slice_range = 0.01, 100

alpha_temp = 2.0
N_u, beta_u = 2**5 - 1, 0.98
N_x3, beta_x3 = 2**8, 15000.0
D_N_u, u_n = cheb_newdom(N_u, beta_u)
x3_n = fourier_dom(N_x3, beta_x3)
u_n_reg = np.copy(u_n)
u_n_reg[-1] = 1.0

N_beta_space = N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n

A_21_y1y2, A_22_y1y2, g_2_y1y2 = readfiledouble("y1y2")
A_21_y3y4, A_22_y3y4, g_2_y3y4 = readfiledouble("y3y4")
A_y_5, g_y5 = readfilesingle("y5")
A_y_6, g_y6 = readfilesingle("y6")
A_21_y7y8, A_22_y7y8, g_2_y7y8 = readfiledouble("y7y8")
read_files = (
    A_21_y1y2,
    A_22_y1y2,
    g_2_y1y2,
    A_21_y3y4,
    A_22_y3y4,
    g_2_y3y4,
    A_y_5,
    g_y5,
    A_y_6,
    g_y6,
    A_21_y7y8,
    A_22_y7y8,
    g_2_y7y8,
)

path = r"C:\Users\matan\PycharmProjects\Black_brane_steady_state_stabilized_Horizon/Saved iteration L 15000 slope 450 A 2_8/"
path = r"saved_data\Saved iteration L 15000 slope 450 A 2_8"

start_time = time.time()
for i in range(len(t_i)):
    time_data = t_i[i], t_f[i], dt, slice_range
    print(f"Current iteration: ({t_i[i], t_f[i]})")
    known_func_t_next_interval = Iterations_RK4(time_data, N_beta_space, known_func_t0, path, read_files)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
print("--- %s hours ---" % ((time.time() - start_time) / 3600))
winsound.Beep(640, 1500)  # Beep at 1000 Hz for 1 second
winsound.Beep(440, 1500)  # Beep at 1000 Hz for 1 second
winsound.Beep(640, 1500)  # Beep at 1000 Hz for 1 second
# %%

# %%
P_0 = -(a1[0, 0] + a1[N_x3 // 2, 0]) / 4
Delta_P = np.abs((a1[0, 0] - a1[N_x3 // 2, 0]) / 4)
delta_P = Delta_P / P_0
print("P_0 = ", P_0)
print("Delta_P = ", Delta_P)
print("delta_P = ", delta_P)

a_t_x3 = known_func_t[0][:, :, -1]  # You can change 0_300 for example.
f_t_x3 = known_func_t[1][:, :, -1]
b_u_t_x3 = known_func_t[2]
N_t_final = known_func_t[2][:, 0, 0].size
#
print(t.shape)
T_3, U_3, X3_3 = np.meshgrid(t / beta_x3, u_n, x3_n / beta_x3)
fig2 = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(
    T_3[0, :N_t_final, len(x3_n) // 2 :],
    X3_3[0, :N_t_final, len(x3_n) // 2 :],
    -a_t_x3[:, len(x3_n) // 2 :] / (2 * P_0),
    cmap=cm.inferno,
)
plt.title(
    "$\\frac{\\langle T^{zz} \\rangle}{P_0}=-\\frac{a(t,x_3)}{2P_0}$", fontdict={"fontweight": "bold", "fontsize": 30}
)
# ax.set_xlabel('$\\frac{t}{L}$')
# ax.set_ylabel('$\\frac{x_3}{L}$')
# ax.set_zlabel('$\frac{-a_4}{P_0}$')
plt.xlabel("$\\frac{t}{L}$")
plt.ylabel("$\\frac{x_3}{L}$")
# plt.zlabel('$\frac{-a_4}{P_0}$')
# ax.set_zlim(0, 1)
plt.show()

fig3 = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(
    T_3[0, :N_t_final, len(x3_n) // 2 :],
    X3_3[0, :N_t_final, len(x3_n) // 2 :],
    f_t_x3[:, len(x3_n) // 2 :] / P_0,
    cmap=cm.inferno,
)
plt.title(
    "$\\frac{\\langle T^{tz} \\rangle}{P_0}=\\frac{f(t,x_3)}{P_0}$", fontdict={"fontweight": "bold", "fontsize": 30}
)
plt.xlabel("$\\frac{t}{L}$")
plt.ylabel("$\\frac{x_3}{L}$")
# ax.set_zlim(0, 1)
plt.show()

fig4 = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(
    T_3[0, :N_t_final, len(x3_n) // 2 :],
    X3_3[0, :N_t_final, len(x3_n) // 2 :],
    b_u_t_x3[:, len(x3_n) // 2 :, -1] / P_0,
    cmap=cm.inferno,
)
plt.title("$b(u=0,t,x_3)$", fontdict={"fontweight": "bold", "fontsize": 30})
plt.xlabel("$\\frac{t}{L}$")
plt.ylabel("$\\frac{x_3}{L}$")
# ax.set_zlim(0, 1)
plt.show()

# %%
# N_u, beta_u = 2**5-1, 0.9
# N_x3, beta_x3 = 2**8, 8000.
# D_N_u, u_n = cheb_newdom(N_u,beta_u)
# x3_n = fourier_dom(N_x3,beta_x3)
# u_n_reg = np.copy(u_n)
# u_n_reg[-1] = 1.
# U, X3   = np.meshgrid(u_n, x3_n)
dt = 0.01

N_beta_space = N_u, N_x3, beta_u, beta_x3, D_N_u, u_n_reg, x3_n

# l_lambda = lambda u,x3: 0*x3
# lam1 = l_lambda(U,X3)
known_func_tpdt = known_func_t[0][-1, :, :], known_func_t[1][-1, :, :], known_func_t[2][-1, :, :], lam1
known_func_t0 = known_func_t[0][-2, :, :], known_func_t[1][-2, :, :], known_func_t[2][-2, :, :], lam1
known_func_tmdt = known_func_t[0][-3, :, :], known_func_t[1][-3, :, :], known_func_t[2][-3, :, :], lam1

b_f = known_func_t[2][-1, :, :]
b = known_func_t[2][-2, :, :]
b_p = known_func_t[2][-3, :, :]

A_21_y1y2, A_22_y1y2, g_2_y1y2 = readfiledouble("y1y2")
A_21_y3y4, A_22_y3y4, g_2_y3y4 = readfiledouble("y3y4")
A_y_5, g_y5 = readfilesingle("y5")
A_y_6, g_y6 = readfilesingle("y6")
A_21_y7y8, A_22_y7y8, g_2_y7y8 = readfiledouble("y7y8")
A_y_9, g_y9 = readfilesingle("y9")
read_files = (
    A_21_y1y2,
    A_22_y1y2,
    g_2_y1y2,
    A_21_y3y4,
    A_22_y3y4,
    g_2_y3y4,
    A_y_5,
    g_y5,
    A_y_6,
    g_y6,
    A_21_y7y8,
    A_22_y7y8,
    g_2_y7y8,
    A_y_9,
    g_y9,
)

y_1_f, y_2_f, y_3_f, y_4_f, y_5_f, y_6_f, y_7_f, y_8_f, y_9_f = solve_y9(known_func_tpdt, N_beta_space, read_files)
y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9 = solve_y9(known_func_t0, N_beta_space, read_files)
y_1_p, y_2_p, y_3_p, y_4_p, y_5_p, y_6_p, y_7_p, y_8_p, y_9_p = solve_y9(known_func_tmdt, N_beta_space, read_files)

Ys_and_b = y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, b

dtb = (b_f - b_p) / (2 * dt)
dty_1 = (y_1_f - y_1_p) / (2 * dt)
dty_7 = (y_7_f - y_7_p) / (2 * dt)
dtY = dtb, dty_1, dty_7

dtdtb = (b_f + b_p - 2 * b) / dt**2
dtdty_1 = (y_1_f + y_1_p - 2 * y_1) / dt**2
dtdtY = dtdtb, dtdty_1

file_path = r"C:\Users\matan\Documents\Numerical solution of gravitational dynamics in asymptotically anti-de Sitter spacetimes\Black brane steady states Speed_up_code_2\Reimann_2Scalar.txt"
file_path = r"Reimann_2Scalar.txt"

# Open and read the file
with open(file_path, "r") as file:
    lines = file.readlines()
for i in range(len(lines)):
    if lines[i].strip() == "Reimannsquared":
        Rsquare_r = lines[i + 1].strip()
Rsquare = parse_mathematica(Rsquare_r)
# %%
RS = kretschmann_scalar(Ys_and_b, dtY, dtdtY, N_beta_space, Rsquare)

fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection="3d")
ax.plot_surface(U, X3, RS, cmap=cm.inferno)
plt.title(
    "Riemann scalar: $R^{\\mu\\nu}_{\\ \\ \\alpha\\beta}R^{\\alpha\\beta}_{\\ \\ \\mu\\nu}$",
    fontdict={"fontweight": "bold", "fontsize": 30},
)
plt.xlabel("$u$")
plt.ylabel("$x_3$")
plt.show()

# %%
# Lambda transformation

alpha_temp = 1.0
N_u, beta_u = 2**6 - 1, 0.9
N_x3, beta_x3 = 2**5, 1000.0
D_N_u, u_n = cheb_newdom(N_u, beta_u)
x3_n = fourier_dom(N_x3, beta_x3)
U, X3 = np.meshgrid(u_n, x3_n)
l = beta_x3 / 4
slope = 0.006
beta = 10.0

# Setting the Riemann problem boundary conditions
P_L, P_R = 0.5, 0.25
vz_L, vz_R = 0.0, -0.25

# Given the pressure and the velocity, we conclude that
a_L, a_R = -(2 / 3) * ((3 + vz_L**2) / (1 - vz_L**2)) * P_L, -(2 / 3) * ((3 + vz_R**2) / (1 - vz_R**2)) * P_R
f_L, f_R = -4 * (vz_L / (1 - vz_L**2)) * P_L, -4 * (vz_R / (1 - vz_R**2)) * P_R
b4_L, b4_R = -(4 / 3) * (vz_L**2 / (1 - vz_L**2)) * P_L, -(4 / 3) * (vz_R**2 / (1 - vz_R**2)) * P_R
print("a_L=", a_L)
print("a_R=", a_R)
a_lambda = lambda u, x3: (a_L + a_R) / 2 + ((a_L - a_R) / 2) * (
    (np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope)))
) / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
f_lambda = lambda u, x3: (f_L + f_R) / 2 + ((f_L - f_R) / 2) * (
    (np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope)))
) / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
b_lambda = lambda u, x3: u * (
    (b4_L + b4_R) / 2
    + ((b4_L - b4_R) / 2)
    * ((np.tanh(beta * np.tanh((x3 + l) * slope))) * (np.tanh(beta * np.tanh((x3 - l) * slope))))
    / ((np.tanh(beta * np.tanh((l) * slope))) * (np.tanh(beta * np.tanh((-l) * slope))))
)

b = b_lambda(U, X3)
f1 = f_lambda(U, X3)
a1 = a_lambda(U, X3)

l1 = (-2 * a1) ** (1 / 4) - 1.0
l2 = -l1

# Coordinate transformation:
N_interpolation = N_u - 5
b_1, u_1 = b_lambda_transformation_fixed_len_two_var(u_n, x3_n.size, N_interpolation, b, l1)
b_2, u_2 = b_lambda_transformation_fixed_len_two_var(u_1, x3_n.size, N_interpolation, b_1, l2)

plt.figure(figsize=(8, 5))
plt.plot(u_n, b[0, :], ".r", label=r"$b(u)$")
plt.plot(u_1, b_1[0, :], ".b", label=r"$\frac{1}{(1-\lambda_1 u)^3}b\left(\frac{u}{1-\lambda_1 u}\right)$")
plt.plot(
    u_2,
    b_2[1, :],
    "yellow",
    label=r"$\frac{1}{(1-(\lambda_1+\lambda_2) u)^3}b\left(\frac{u}{1-(\lambda_1+\lambda_2) u}\right)$",
)
# plt.axvline(x = u_cut, color = 'black', linestyle='--', label = 'axvline - full height')
plt.xlabel("$u$")
plt.ylabel("$b(u)$")
plt.legend()
plt.grid(True)
plt.show()


# %%
# Two variables u and x:
N_u, u_cut = 64, 1.0
# u = np.linspace(0., u_max, N_u) # start at 0., end at u0 and have N points
D_N_u, u_n = cheb_newdom(N_u, 1 / u_cut)
print(u_n.size)
x = np.linspace(0, 1, 100)

gamma = 5.0
w0 = 0.15
b_lambda = lambda u_var, x_var: gamma * u_var * np.exp(-(u_var**2) / w0**2)
lam_1_lambda = lambda u_var, x_var: 0 * x_var + 0.5
U, X = np.meshgrid(u_n, x)
b = b_lambda(U, X)
lam_1 = lam_1_lambda(U, X)
lam_2 = -lam_1

# Coordinate transformation:
N_interpolation = 60
b_1, u_1 = b_lambda_transformation_fixed_len_two_var(u_n, x.size, N_interpolation, b, lam_1)
b_2, u_2 = b_lambda_transformation_fixed_len_two_var(u_1, x.size, N_interpolation, b_1, lam_2)

plt.figure(figsize=(8, 5))
plt.plot(u_n, b[0, :], ".r", label=r"$b(u)$")
plt.plot(u_1, b_1[0, :], ".b", label=r"$\frac{1}{(1-\lambda_1 u)^3}b\left(\frac{u}{1-\lambda_1 u}\right)$")
plt.plot(
    u_2,
    b_2[1, :],
    "yellow",
    label=r"$\frac{1}{(1-(\lambda_1+\lambda_2) u)^3}b\left(\frac{u}{1-(\lambda_1+\lambda_2) u}\right)$",
)
# plt.axvline(x = u_cut, color = 'black', linestyle='--', label = 'axvline - full height')
plt.xlabel("$u$")
plt.ylabel("$b(u)$")
plt.legend()
plt.grid(True)
plt.show()
