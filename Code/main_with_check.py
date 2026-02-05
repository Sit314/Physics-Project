# %%
import os

import dill as pickle
import sympy as sp
from einsteinpy.symbolic import EinsteinTensor, MetricTensor
from IPython.display import Math, display

# Import from our modules
from src.config import A, B_dot, F, Sigma, Sigma_dot, coords, metric_array, r
from src.display import display_ordered_equation
from src.printing import dot_latex
from src.transform import apply_dot_operator, clean_equation

# ==========================================
# 1. Calculation & Caching
# ==========================================
CACHE_FILE = "einstein_tensor_dill.pkl"


def get_einstein_tensor():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Calculating Tensor...")
    G = MetricTensor(metric_array, coords)
    G_uv_list = EinsteinTensor.from_metric(G).tensor().tolist()

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(G_uv_list, f)
    return G_uv_list


# ==========================================
# 2. Helper Functions
# ==========================================


def check_simplification(tensor_list, substitutions, step_name):
    """
    Applies the accumulated substitutions to the Einstein Tensor
    and prints non-zero components to verify simplification.
    """
    print(f"\n" + "-" * 60)
    print(f"Checking Tensor Simplification after {step_name}")
    print("-" * 60)

    remaining_equations = 0

    for i in range(4):
        for j in range(i, 4):
            component = tensor_list[i][j]
            if component != 0:
                # 1. Apply dot operator to get standard notation
                transformed = apply_dot_operator(component)

                # 2. Apply all accumulated substitutions
                for target, replacement in substitutions:
                    transformed = transformed.subs(target, replacement)

                # --- FIX: Safety Check ---
                # If substitution resulted in exact 0, skip clean_equation
                # to avoid "ComputationFailed: primitive(0)"
                if transformed == 0:
                    continue

                # Optional: specific check for complex cancellations
                # Only use simplify if it's not clearly zero to save time
                if sp.simplify(transformed) == 0:
                    continue
                # -------------------------

                # 3. Clean and Simplify
                final_eq = clean_equation(transformed)
                final_eq = sp.simplify(final_eq)

                if final_eq != 0:
                    remaining_equations += 1
                    label = rf"G_{{{i}{j}}}"
                    latex_str = dot_latex(final_eq)
                    display(Math(rf"\text{{Remaining }} {label}: \\\\ {latex_str} = 0"))

    if remaining_equations == 0:
        print(f">>> PERFECT SIMPLIFICATION: All components are zero after {step_name}.")
    else:
        print(f">>> {remaining_equations} non-zero components remaining.")


# ==========================================
# 3. Main Execution Flow
# ==========================================

G_uv_list = get_einstein_tensor()
accumulated_subs = []  # List of tuples (target, substitution)

print("=" * 26 + "\nEinstein Tensor Components\n" + "=" * 26)
open("equations.txt", "w").close()  # Reset file

# --- Initial Print of All Components ---
print("Initial State (No substitutions):")
for i in range(4):
    for j in range(i, 4):
        component = G_uv_list[i][j]
        if component != 0:
            transformed = apply_dot_operator(component)
            final_eq = clean_equation(transformed)
            label = rf"G_{{{i}{j}}}"
            latex_str = dot_latex(final_eq)
            display(Math(rf"{label}: \\\\ {latex_str} = 0"))
            print("-" * 180)


# --- Specific Analysis Steps ---

# (2.8a) G_22 -> Sigma
print("\n(2.8a) Extracting Σ Equation from G_22 (G_rr)")
eq_Sigma = clean_equation(apply_dot_operator(G_uv_list[2][2]))
display_ordered_equation(eq_Sigma, Sigma, label="(2.8a)")

# Solve for Sigma'' and add to subs
sol_sigma = sp.solve(eq_Sigma, Sigma.diff(r, r))
if sol_sigma:
    accumulated_subs.append((Sigma.diff(r, r), sol_sigma[0]))
    check_simplification(G_uv_list, accumulated_subs, "(2.8a) Sigma_rr substitution")


# (2.8b) G_12 -> F
print("\n(2.8b) Extracting F Equation from G_12 (G_zr)")
eq_F = -clean_equation(apply_dot_operator(G_uv_list[1][2]))
display_ordered_equation(eq_F, F, label="(2.8b)")

# Solve for F' and add to subs
sol_F = sp.solve(eq_F, F.diff(r))
if sol_F:
    accumulated_subs.append((F.diff(r), sol_F[0]))
    check_simplification(G_uv_list, accumulated_subs, "(2.8b) F_r substitution")


# (2.8c) G_02 -> Sigma_dot
print("\n(2.8c) Extracting Σ_dot Equation")
eq_Sigma_dot_raw = clean_equation(apply_dot_operator(G_uv_list[0][2])) / 2
# We must apply previous substitutions to the equation itself before solving
for target, replacement in accumulated_subs:
    eq_Sigma_dot_raw = eq_Sigma_dot_raw.subs(target, replacement)

eq_Sigma_dot = clean_equation(eq_Sigma_dot_raw)
display_ordered_equation(eq_Sigma_dot, Sigma_dot, label="(2.8c)")

# Solve for Sigma_dot'
# Note: Since Sigma_dot is a function of t and r, we look for Diff(Sigma(t,r), t, r)
# But apply_dot_operator turns derivative(Sigma, t) into Sigma_dot atom.
# So we look for derivative of Sigma_dot w.r.t r
sol_sigma_dot = sp.solve(eq_Sigma_dot, Sigma_dot.diff(r))
if sol_sigma_dot:
    accumulated_subs.append((Sigma_dot.diff(r), sol_sigma_dot[0]))
    check_simplification(G_uv_list, accumulated_subs, "(2.8c) Sigma_dot_r substitution")


# (2.8d) G_11 and G_33 -> B_dot
print("\n(2.8d) Extracting B_dot Equation")
term_1 = clean_equation(G_uv_list[1][1])
term_2 = clean_equation(G_uv_list[3][3])
eq_B_dot_raw = clean_equation(apply_dot_operator((term_1 - Sigma * term_2) / 2))

# Apply accumulated subs
for target, replacement in accumulated_subs:
    eq_B_dot_raw = eq_B_dot_raw.subs(target, replacement)

eq_B_dot = clean_equation(eq_B_dot_raw)
display_ordered_equation(eq_B_dot, B_dot, label="(2.8d)")

# Solve for B_dot
sol_b_dot = sp.solve(eq_B_dot, B_dot)
if sol_b_dot:
    accumulated_subs.append((B_dot, sol_b_dot[0]))
    check_simplification(G_uv_list, accumulated_subs, "(2.8d) B_dot substitution")


# (2.8e) G_11 -> A
print("\n(2.8e) Extracting A Equation from 1/2 G_11")
eq_A_raw = clean_equation(apply_dot_operator(G_uv_list[1][1])) / 2

# Apply accumulated subs
for target, replacement in accumulated_subs:
    eq_A_raw = eq_A_raw.subs(target, replacement)

eq_A = clean_equation(eq_A_raw)
display_ordered_equation(eq_A, A, label="(2.8e)")

# Solve for A''
sol_A = sp.solve(eq_A, A.diff(r, r))
if sol_A:
    accumulated_subs.append((A.diff(r, r), sol_A[0]))
    check_simplification(G_uv_list, accumulated_subs, "(2.8e) A_rr substitution")

# %%
