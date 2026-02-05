from My_PDE_Horizon import fourier_dom
from solve_main_equations import evolve_a1_f1, initial_conditions_duality_paper, plot_3d_surfaces

# Parameters from the paper
lam = 1.0
A0 = 100.0
alpha = 0.8657
beta = 0.5

N_x3 = 256
N_r = 33
L = 1000.0
dt = 0.05
t_max = 6.0 * lam

fields, a1, f1 = initial_conditions_duality_paper(N_x3, N_r, L, A0, alpha, beta, lam)

# times, a1_hist, f1_hist, b3_hist = evolve_a1_f1(fields, a1, f1, z_length=L, lam=lam, t_max=t_max, dt=dt, r_min=1.0, r_max=50.0)
times, a1_hist, f1_hist, b3_hist = evolve_a1_f1(
    fields, a1, f1, z_length=L, lam=lam, t_max=t_max, dt=dt, r_min=1.0, r_max=50.0, filter_fields=True, clip_b=20.0
)

z_grid = fourier_dom(N_x3, L)
plot_3d_surfaces(times, z_grid, lam, a1_hist, f1_hist)
