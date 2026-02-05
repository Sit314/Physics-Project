import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from main import get_equations
from mpl_toolkits.mplot3d import Axes3D
from My_PDE_Horizon import cheb, fourier_derivative, fourier_derivative_two_variables, fourier_dom, low_pass_filter, low_pass_filter_with_u
from src.config import A, B, B_dot, F, Sigma, Sigma_dot, r, z


def cheb_grid(n_points, r_min, r_max):
    d_x, x = cheb(n_points - 1)
    r_grid = 0.5 * (r_max - r_min) * x + 0.5 * (r_max + r_min)
    d_r = 2.0 / (r_max - r_min) * d_x
    d_rr = d_r @ d_r
    return d_r, d_rr, r_grid


def _interp_to_u(r_grid, values, u_grid):
    u_safe = np.clip(u_grid, 1e-6, None)
    r_sorted_idx = np.argsort(r_grid)
    r_sorted = r_grid[r_sorted_idx]
    values_sorted = values[:, r_sorted_idx]
    r_from_u = 1.0 / u_safe

    out = np.zeros((values.shape[0], u_grid.size))
    for i_z in range(values.shape[0]):
        out[i_z, :] = np.interp(r_from_u, r_sorted, values_sorted[i_z, :])
    return out


def solve_sigma_u(b_field, b3, d_r, r_grid, u_min=1e-6, u_max=1.0):
    n_z, n_r = b_field.shape
    d_u, d_uu, u_grid = cheb_grid(n_r, u_min, u_max)

    b_r = (d_r @ b_field.T).T
    b_r_u = _interp_to_u(r_grid, b_r, u_grid)

    sigma_u = np.zeros((n_z, n_r))
    boundary_index = n_r - 1
    u_boundary = u_grid[boundary_index]

    for i_z in range(n_z):
        sigma_val = (1.0 / u_boundary) - (3.0 * b3[i_z] * (u_boundary**5)) / 40.0
        sigma_u_val = -(1.0 / (u_boundary**2)) - (3.0 * b3[i_z] * (u_boundary**4)) / 8.0

        a_22 = 4.0 * (u_grid**4)
        a_21 = 8.0 * (u_grid**3)
        a_0 = b_r_u[i_z, :] ** 2
        rhs = np.zeros_like(u_grid)

        sigma_u[i_z, :] = solve_linear_bvp(a_22, a_21, a_0, rhs, d_u, d_uu, sigma_val, sigma_u_val, boundary_index=boundary_index)

    r_from_u = 1.0 / u_grid
    r_sorted_idx = np.argsort(r_from_u)
    r_sorted = r_from_u[r_sorted_idx]

    sigma_r = np.zeros((n_z, n_r))
    for i_z in range(n_z):
        sigma_sorted = sigma_u[i_z, r_sorted_idx]
        sigma_r[i_z, :] = np.interp(r_grid, r_sorted, sigma_sorted)

    return sigma_r


def extract_linear_ode(eq, target_func, var=r):
    expr = sp.expand(eq)
    f_rr = sp.Derivative(target_func, var, var)
    f_r = sp.Derivative(target_func, var)

    a_22 = sp.simplify(expr.coeff(f_rr))
    remaining = expr - a_22 * f_rr

    a_21 = sp.simplify(remaining.coeff(f_r))
    remaining = remaining - a_21 * f_r

    a_0 = sp.simplify(remaining.coeff(target_func))
    rhs = sp.simplify(-(remaining - a_0 * target_func))

    return a_22, a_21, a_0, rhs


def _symbol_for_derivative(deriv):
    base_name = deriv.args[0].func.__name__
    suffix = []
    for var, count in deriv.variable_count:
        suffix.extend([str(var)] * count)
    if suffix:
        return sp.Symbol(base_name + "_" + "".join(suffix))
    return sp.Symbol(base_name)


def _replace_functions(expr):
    replacements = {}
    allowed_funcs = {
        A.func,
        B.func,
        B_dot.func,
        F.func,
        Sigma.func,
        Sigma_dot.func,
    }
    for deriv in expr.atoms(sp.Derivative):
        replacements[deriv] = _symbol_for_derivative(deriv)

    for func in expr.atoms(sp.Function):
        if func.func in allowed_funcs:
            name = func.func.__name__
            replacements[func] = sp.Symbol(name)

    replaced = expr.xreplace(replacements)
    symbols = sorted({sym for sym in replacements.values() if isinstance(sym, sp.Symbol)}, key=lambda s: s.name)
    return replaced, symbols


def prepare_coeff_functions(eq, target_func, var=r):
    a_22, a_21, a_0, rhs = extract_linear_ode(eq, target_func, var=var)

    a_22_sym, symbols_22 = _replace_functions(a_22)
    a_21_sym, symbols_21 = _replace_functions(a_21)
    a_0_sym, symbols_0 = _replace_functions(a_0)
    rhs_sym, symbols_rhs = _replace_functions(rhs)

    symbol_set = {s for s in symbols_22 + symbols_21 + symbols_0 + symbols_rhs}
    symbols = sorted(symbol_set, key=lambda s: s.name)

    func_a22 = sp.lambdify(symbols, a_22_sym, "numpy")
    func_a21 = sp.lambdify(symbols, a_21_sym, "numpy")
    func_a0 = sp.lambdify(symbols, a_0_sym, "numpy")
    func_rhs = sp.lambdify(symbols, rhs_sym, "numpy")

    return {
        "symbols": symbols,
        "a_22": func_a22,
        "a_21": func_a21,
        "a_0": func_a0,
        "rhs": func_rhs,
    }


def _compute_r_derivatives(field, d_r, d_rr):
    f_r = (d_r @ field.T).T
    f_rr = (d_rr @ field.T).T
    f_rrr = (d_r @ f_rr.T).T
    return f_r, f_rr, f_rrr


def _compute_z_derivatives(field, z_length):
    n_z = field.shape[0]
    n_r = field.shape[1]
    par = z_length, n_r - 1
    f_z = fourier_derivative_two_variables(field, par)
    f_zz = fourier_derivative_two_variables(f_z, par)
    return f_z, f_zz


def build_field_data(fields, d_r, d_rr, r_grid, z_length=None):
    data = {}
    n_z = next(iter(fields.values())).shape[0]
    data["r"] = np.tile(r_grid, (n_z, 1))

    for name, field in fields.items():
        data[name] = field

        f_r, f_rr, f_rrr = _compute_r_derivatives(field, d_r, d_rr)
        data[f"{name}_r"] = f_r
        data[f"{name}_rr"] = f_rr
        data[f"{name}_rrr"] = f_rrr

        if z_length is not None:
            f_z, f_zz = _compute_z_derivatives(field, z_length)
            data[f"{name}_z"] = f_z
            data[f"{name}_zz"] = f_zz
            data[f"{name}_rz"] = (d_r @ f_z.T).T
            data[f"{name}_rrz"] = (d_rr @ f_z.T).T

    return data


def evaluate_coefficients(coeff_info, values_by_name, z_index):
    args = []
    for sym in coeff_info["symbols"]:
        name = sym.name
        if name not in values_by_name:
            raise KeyError(f"Missing symbol data for {name}")
        args.append(values_by_name[name][z_index, :])

    a_22 = coeff_info["a_22"](*args)
    a_21 = coeff_info["a_21"](*args)
    a_0 = coeff_info["a_0"](*args)
    rhs = coeff_info["rhs"](*args)

    return a_22, a_21, a_0, rhs


def _apply_value_bc(matrix, rhs, row, col, value):
    matrix[row, :] = 0.0
    matrix[row, col] = 1.0
    rhs[row] = value


def _apply_derivative_bc(matrix, rhs, row, d_r_row, value):
    matrix[row, :] = d_r_row
    rhs[row] = value


def solve_linear_bvp(a_22, a_21, a_0, rhs, d_r, d_rr, bc_value, bc_derivative, boundary_index=0):
    n = d_r.shape[0]
    a_22 = np.asarray(a_22)
    a_21 = np.asarray(a_21)
    a_0 = np.asarray(a_0)
    rhs = np.asarray(rhs)

    if a_22.ndim == 0:
        a_22 = np.full(n, a_22)
    if a_21.ndim == 0:
        a_21 = np.full(n, a_21)
    if a_0.ndim == 0:
        a_0 = np.full(n, a_0)
    if rhs.ndim == 0:
        rhs = np.full(n, rhs)

    mat = (np.diag(a_22) @ d_rr + np.diag(a_21) @ d_r + np.diag(a_0)).astype(float)
    vec = rhs.astype(float, copy=True)

    _apply_value_bc(mat, vec, 0, boundary_index, bc_value)
    _apply_derivative_bc(mat, vec, 1, d_r[boundary_index, :], bc_derivative)

    try:
        sol = np.linalg.solve(mat, vec)
    except np.linalg.LinAlgError:
        sol, _, _, _ = np.linalg.lstsq(mat, vec, rcond=None)
    return sol


def estimate_b3_from_boundary(b_field, r_boundary):
    return b_field[:, 0] * (r_boundary**3)


def bc_sigma(r_boundary, b3):
    # Asymptotic AdS boundary conditions with xi=0.
    sigma_val = r_boundary - (b3 / (40.0 * (r_boundary**5)))
    sigma_r = 1.0 + (b3 / (8.0 * (r_boundary**6)))
    return sigma_val, sigma_r


def bc_f(r_boundary, f1, dz_b3):
    f_val = (f1 / r_boundary) + (3.0 * dz_b3) / (4.0 * (r_boundary**2))
    f_r = (-f1 / (r_boundary**2)) - (3.0 * dz_b3) / (2.0 * (r_boundary**3))
    return f_val, f_r


def bc_sigma_dot(r_boundary, a1):
    sigma_dot_val = 0.5 * (r_boundary**2) + a1 / r_boundary
    sigma_dot_r = r_boundary - a1 / (r_boundary**2)
    return sigma_dot_val, sigma_dot_r


def bc_b_dot(r_boundary, b3):
    b_dot_val = -1.5 * b3 / (r_boundary**2)
    b_dot_r = 3.0 * b3 / (r_boundary**3)
    return b_dot_val, b_dot_r


def bc_a(r_boundary, a1):
    a_val = 0.5 * (r_boundary**2) + a1 / r_boundary
    a_r = r_boundary - a1 / (r_boundary**2)
    return a_val, a_r


def solve_equations(fields, a1, f1, z_length, r_min=1.0, r_max=50.0):
    """Solve 2.8a-2.8e using boundary data a1, f1 and current B field."""
    eqs = get_equations()

    d_r, d_rr, r_grid = cheb_grid(fields["B"].shape[1], r_min, r_max)
    boundary_index = 0
    r_boundary = r_grid[boundary_index]

    b3 = estimate_b3_from_boundary(fields["B"], r_boundary)
    dz_b3 = fourier_derivative(b3, z_length)

    coeff_sigma = prepare_coeff_functions(eqs["eq_Sigma"], Sigma)
    coeff_f = prepare_coeff_functions(eqs["eq_F"], F)
    coeff_sigma_dot = prepare_coeff_functions(eqs["eq_Sigma_dot"], Sigma_dot)
    coeff_b_dot = prepare_coeff_functions(eqs["eq_B_dot"], B_dot)
    coeff_a = prepare_coeff_functions(eqs["eq_A"], A)

    n_z = fields["B"].shape[0]
    sigma = np.zeros_like(fields["B"])
    f = np.zeros_like(fields["B"])
    sigma_dot = np.zeros_like(fields["B"])
    b_dot = np.zeros_like(fields["B"])
    a = np.zeros_like(fields["B"])

    sigma = solve_sigma_u(fields["B"], b3, d_r, r_grid)

    data_f = build_field_data({"B": fields["B"], "Sigma": sigma}, d_r, d_rr, r_grid, z_length=z_length)
    for i_z in range(n_z):
        f_val, f_r = bc_f(r_boundary, f1[i_z], dz_b3[i_z])
        a_22, a_21, a_0, rhs = evaluate_coefficients(coeff_f, data_f, i_z)
        f[i_z, :] = solve_linear_bvp(a_22, a_21, a_0, rhs, d_r, d_rr, f_val, f_r, boundary_index=boundary_index)

    data_sigma_dot = build_field_data({"B": fields["B"], "Sigma": sigma, "F": f}, d_r, d_rr, r_grid, z_length=z_length)
    for i_z in range(n_z):
        sigma_dot_val, sigma_dot_r = bc_sigma_dot(r_boundary, a1[i_z])
        a_22, a_21, a_0, rhs = evaluate_coefficients(coeff_sigma_dot, data_sigma_dot, i_z)
        sigma_dot[i_z, :] = solve_linear_bvp(a_22, a_21, a_0, rhs, d_r, d_rr, sigma_dot_val, sigma_dot_r, boundary_index=boundary_index)

    data_b_dot = build_field_data({"B": fields["B"], "Sigma": sigma, "F": f, "Sigma_dot": sigma_dot}, d_r, d_rr, r_grid, z_length=z_length)
    for i_z in range(n_z):
        b_dot_val, b_dot_r = bc_b_dot(r_boundary, b3[i_z])
        a_22, a_21, a_0, rhs = evaluate_coefficients(coeff_b_dot, data_b_dot, i_z)
        b_dot[i_z, :] = solve_linear_bvp(a_22, a_21, a_0, rhs, d_r, d_rr, b_dot_val, b_dot_r, boundary_index=boundary_index)

    data_a = build_field_data(
        {"B": fields["B"], "Sigma": sigma, "F": f, "Sigma_dot": sigma_dot, "B_dot": b_dot}, d_r, d_rr, r_grid, z_length=z_length
    )
    for i_z in range(n_z):
        a_val, a_r = bc_a(r_boundary, a1[i_z])
        a_22, a_21, a_0, rhs = evaluate_coefficients(coeff_a, data_a, i_z)
        a[i_z, :] = solve_linear_bvp(a_22, a_21, a_0, rhs, d_r, d_rr, a_val, a_r, boundary_index=boundary_index)

    return {
        "r_grid": r_grid,
        "z_grid": fourier_dom(fields["B"].shape[0], z_length),
        "Sigma": sigma,
        "F": f,
        "Sigma_dot": sigma_dot,
        "B_dot": b_dot,
        "A": a,
        "b3": b3,
        "dz_b3": dz_b3,
    }


def initial_conditions_duality_paper(n_z, n_r, z_length, a0, alpha, beta, lam):
    """Initial conditions from eq. (2.17) in מאמר דואליות.pdf (xi = 0)."""
    z_grid = fourier_dom(n_z, z_length)
    a1 = -a0 * (1.0 - alpha * np.tanh(beta * np.tanh(z_grid / lam)))
    f1 = np.zeros_like(a1)
    b_field = np.zeros((n_z, n_r))
    return {"B": b_field}, a1, f1


def evolve_a1_f1(
    fields,
    a1,
    f1,
    z_length,
    lam,
    t_max,
    dt,
    r_min=1.0,
    r_max=50.0,
    filter_fields=True,
    clip_b=20.0,
):
    times = np.arange(0.0, t_max + 0.5 * dt, dt)
    n_t = times.size
    n_z = a1.size
    n_r = fields["B"].shape[1]
    d_r, _, r_grid = cheb_grid(n_r, r_min, r_max)

    a1_hist = np.zeros((n_t, n_z))
    f1_hist = np.zeros((n_t, n_z))
    b3_hist = np.zeros((n_t, n_z))

    a1_hist[0, :] = a1
    f1_hist[0, :] = f1

    for i_t in range(1, n_t):
        if clip_b is not None:
            fields["B"] = np.clip(fields["B"], -clip_b, clip_b)

        result = solve_equations(fields, a1, f1, z_length=z_length, r_min=r_min, r_max=r_max)

        b3 = result["b3"]
        dz_a1 = fourier_derivative(a1, z_length)
        dz_f1 = fourier_derivative(f1, z_length)
        dz_b3 = result["dz_b3"]

        a1_t = 0.75 * dz_f1
        f1_t = (2.0 / 3.0) * dz_a1 + dz_b3

        a1 = a1 + dt * a1_t
        f1 = f1 + dt * f1_t

        # Convert dot derivative to partial t derivative: d_t B = B_dot - A * d_r B
        b_r = (d_r @ fields["B"].T).T
        b_t = result["B_dot"] - result["A"] * b_r
        fields["B"] = fields["B"] + dt * b_t

        if filter_fields:
            a1 = low_pass_filter(a1, z_length)
            f1 = low_pass_filter(f1, z_length)
            fields["B"] = low_pass_filter_with_u(fields["B"], z_length, r_grid)

        a1_hist[i_t, :] = a1
        f1_hist[i_t, :] = f1
        b3_hist[i_t, :] = b3

    return times, a1_hist, f1_hist, b3_hist


def plot_3d_surfaces(times, z_grid, lam, a1_hist, f1_hist, p0=None):
    if p0 is None:
        p0 = -(a1_hist[0, 0] + a1_hist[0, a1_hist.shape[1] // 2]) / 4.0

    z_scaled = np.tanh(z_grid / (10.0 * lam))
    t_scaled = times / lam

    j_over_p0 = (1.5 * f1_hist) / p0
    p_over_p0 = (-0.5 * a1_hist) / p0

    z_mesh, t_mesh = np.meshgrid(z_scaled, t_scaled)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(z_mesh, t_mesh, j_over_p0, cmap="viridis", linewidth=0, antialiased=True)
    ax1.set_xlabel("tanh(z / (10 lambda))")
    ax1.set_ylabel("t / lambda")
    ax1.set_zlabel("J / P0")
    ax1.set_xlim(-1.0, 1.0)
    ax1.set_ylim(0.0, 6.0)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(z_mesh, t_mesh, p_over_p0, cmap="plasma", linewidth=0, antialiased=True)
    ax2.set_xlabel("tanh(z / (10 lambda))")
    ax2.set_ylabel("t / lambda")
    ax2.set_zlabel("P / P0")
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_ylim(0.0, 6.0)

    fig.tight_layout()
    plt.show()


def plot_results(result, r_index=0, title_suffix=""):
    z_grid = result["z_grid"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    axes[0].plot(z_grid, result["A"][:, r_index])
    axes[0].set_ylabel("A")

    axes[1].plot(z_grid, result["F"][:, r_index])
    axes[1].set_ylabel("F")

    axes[2].plot(z_grid, result["Sigma"][:, r_index])
    axes[2].set_ylabel("Sigma")

    axes[3].plot(z_grid, result["B_dot"][:, r_index])
    axes[3].set_ylabel("B_dot")

    axes[4].plot(z_grid, result["Sigma_dot"][:, r_index])
    axes[4].set_ylabel("Sigma_dot")
    axes[4].set_xlabel("z")

    axes[5].plot(z_grid, result["b3"])
    axes[5].set_ylabel("b3")
    axes[5].set_xlabel("z")

    if title_suffix:
        fig.suptitle(title_suffix)
    fig.tight_layout()
    plt.show()
