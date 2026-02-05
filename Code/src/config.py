import sympy as sp

# 1. Define Symbols
t, z, r, x = sp.symbols("t z r x")
coords = [t, z, r, x]

# 2. Define Functions
A = sp.Function("A")(t, z, r)
F = sp.Function("F")(t, z, r)
B = sp.Function("B")(t, z, r)
Sigma = sp.Function("Sigma")(t, z, r)

# 3. Define Derivatives (for substitution)
A_dot = sp.Function("A_dot")(t, z, r)
B_dot = sp.Function("B_dot")(t, z, r)
F_dot = sp.Function("F_dot")(t, z, r)
Sigma_dot = sp.Function("Sigma_dot")(t, z, r)

A_ddot = sp.Function("A_ddot")(t, z, r)
B_ddot = sp.Function("B_ddot")(t, z, r)
F_ddot = sp.Function("F_ddot")(t, z, r)
Sigma_ddot = sp.Function("Sigma_ddot")(t, z, r)

# 4. Mappings
# Used by the transform logic to know what relates to what
DOT_MAP = {"A": (A, A_dot, A_ddot), "B": (B, B_dot, B_ddot), "F": (F, F_dot, F_ddot), "Sigma": (Sigma, Sigma_dot, Sigma_ddot)}

# 5. The Metric
metric_array = [[-2 * A, -F, 1, 0], [-F, Sigma**2 * sp.exp(-B), 0, 0], [1, 0, 0, 0], [0, 0, 0, Sigma**2 * sp.exp(B)]]
