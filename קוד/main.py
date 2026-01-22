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
# 2. Main Execution Flow
# ==========================================

G_uv_list = get_einstein_tensor()

print("=" * 26 + "\nEinstein Tensor Components\n" + "=" * 26)
open("equations.txt", "w").close()  # Reset file

# --- Print All Components ---
for i in range(4):
    for j in range(i, 4):
        component = G_uv_list[i][j]
        if component != 0:
            transformed = apply_dot_operator(component)
            final_eq = clean_equation(transformed)

            label = rf"G_{{{i}{j}}}"
            latex_str = dot_latex(final_eq)

            display(Math(rf"{label}: \\\\ {latex_str} = 0"))
            # with open("equations.txt", "a", encoding="utf-8") as f:
            #     f.write(f"$${latex_str} = 0$$\n")

            # display_ordered_equation(final_eq, Sigma_ddot, label=label)
            print("-" * 180)

# --- Specific Analysis Steps ---

# (2.8a) G_22 -> Sigma
print("\n(2.8a) Extracting Σ Equation from G_22 (G_rr)")
eq_Sigma = clean_equation(apply_dot_operator(G_uv_list[2][2]))
display_ordered_equation(eq_Sigma, Sigma, label="(2.8a)")

# (2.8b) G_12 -> F
print("\n(2.8b) Extracting F Equation from G_12 (G_zr)")
eq_F = -clean_equation(apply_dot_operator(G_uv_list[1][2]))
display_ordered_equation(eq_F, F, label="(2.8b)")

# (2.8c) G_02 -> Sigma_dot (Requires substitution from 2.8a)
print("\n(2.8c) Extracting Σ_dot Equation")
eq_Sigma_dot = clean_equation(apply_dot_operator(G_uv_list[0][2])) / 2
d2_sigma_dr2 = sp.solve(eq_Sigma, Sigma.diff(r, r))[0]
eq_Sigma_dot = eq_Sigma_dot.subs(Sigma.diff(r, r), d2_sigma_dr2)
display_ordered_equation(eq_Sigma_dot, Sigma_dot, label="(2.8c)")

# (2.8d) G_11 and G_33 -> B_dot
print("\n(2.8d) Extracting B_dot Equation")
# G_11 - Sigma * G_33
term_1 = clean_equation(G_uv_list[1][1])
term_2 = clean_equation(G_uv_list[3][3])
eq_B_dot = clean_equation(apply_dot_operator((term_1 - Sigma * term_2) / 2))
display_ordered_equation(eq_B_dot, B_dot, label="(2.8d)")

# (2.8e) G_11 -> A
print("\n(2.8e) Extracting A Equation from 1/2 G_11")
eq_A = clean_equation(apply_dot_operator(G_uv_list[1][1])) / 2
display_ordered_equation(eq_A, A, label="(2.8e)")

# %%
