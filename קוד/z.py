# %%
import sympy as sp
from einsteinpy.symbolic import MetricTensor, EinsteinTensor
import os
import dill as pickle
from IPython.display import display, Math
from sympy.printing.latex import LatexPrinter


# ==========================================
# 1. Dot Printer (Dark Mode & Strict Scope)
# ==========================================
class DotPrinter(LatexPrinter):
    def _print_Function(self, expr, exp=None):
        func_name = expr.func.__name__

        # 1. Identify Base Name and Derivative Level
        if func_name.endswith("_ddot"):
            base = func_name.replace("_ddot", "")
            deriv_type = "ddot"
        elif func_name.endswith("_dot"):
            base = func_name.replace("_dot", "")
            deriv_type = "dot"
        else:
            base = func_name
            deriv_type = "base"

        # 2. Define Color Maps (Optimized for Dark Backgrounds)
        # Lighter/Pastel for Base/Dot, slightly deeper for Ddot to maintain contrast
        color_map = {
            # A: Salmon / Pink / Deep Red
            "A": {"base": "#FF6666", "dot": "#FFAAAA", "ddot": "#CC0000"},
            # B: Sky Blue / Pale Blue / Medium Blue
            "B": {"base": "#44AAFF", "dot": "#88CCFF", "ddot": "#0066CC"},
            # F: Lime Green / Pale Green / Green
            "F": {"base": "#32CD32", "dot": "#98FB98", "ddot": "#008800"},
            # Sigma: Gold / Pale Yellow / Orange
            "Sigma": {"base": "#FFC125", "dot": "#FFE082", "ddot": "#FF8C00"},
        }

        base_latex = r"\Sigma" if base == "Sigma" else base

        # 3. Generate LaTeX with Strict Scoping
        if base in color_map:
            colors = color_map[base]

            # Determine the inner symbol string
            if deriv_type == "ddot":
                # \ddot{A}
                symbol_str = rf"\ddot{{{base_latex}}}"
                color = colors["ddot"]
            elif deriv_type == "dot":
                # \dot{A}
                symbol_str = rf"\dot{{{base_latex}}}"
                color = colors["dot"]
            else:
                # A
                symbol_str = base_latex
                color = colors["base"]

            # Wrap ONLY the symbol in the color command: {\color{code} symbol}
            # This prevents the color from bleeding into following symbols or exponents
            colored_symbol = rf"{{\color{{{color}}} {symbol_str}}}"

            # Append the exponent outside the colored block
            if exp:
                return rf"{colored_symbol}^{{{exp}}}"

            return colored_symbol

        return super()._print_Function(expr, exp)

    def _print_Derivative(self, expr):
        function = expr.args[0]
        func_str = self._print(function)
        indices_str = ""
        for var, count in expr.variable_count:
            indices_str += self._print(var) * count
        return rf"\partial_{{{indices_str}}} {func_str}"


def dot_latex(expr):
    return DotPrinter().doprint(expr)


# ==========================================
# 2. Setup Variables
# ==========================================
t, z, r, x = sp.symbols("t z r x")
coords = [t, z, r, x]

A = sp.Function("A")(t, z, r)
F = sp.Function("F")(t, z, r)
B = sp.Function("B")(t, z, r)
Sigma = sp.Function("Sigma")(t, z, r)

A_dot = sp.Function("A_dot")(t, z, r)
B_dot = sp.Function("B_dot")(t, z, r)
F_dot = sp.Function("F_dot")(t, z, r)
Sigma_dot = sp.Function("Sigma_dot")(t, z, r)

A_ddot = sp.Function("A_ddot")(t, z, r)
B_ddot = sp.Function("B_ddot")(t, z, r)
F_ddot = sp.Function("F_ddot")(t, z, r)
Sigma_ddot = sp.Function("Sigma_ddot")(t, z, r)

dot_map = {"A": (A, A_dot, A_ddot), "B": (B, B_dot, B_ddot), "F": (F, F_dot, F_ddot), "Sigma": (Sigma, Sigma_dot, Sigma_ddot)}

metric_array = [[-2 * A, -F, 1, 0], [-F, Sigma**2 * sp.exp(-B), 0, 0], [1, 0, 0, 0], [0, 0, 0, Sigma**2 * sp.exp(B)]]

# ==========================================
# 3. Calculation
# ==========================================
cache_file = "einstein_tensor_dill.pkl"
if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        G_uv_list = pickle.load(f)
else:
    print("Calculating Tensor...")
    g = MetricTensor(metric_array, coords)
    G_uv_list = EinsteinTensor.from_metric(g).tensor().tolist()
    with open(cache_file, "wb") as f:
        pickle.dump(G_uv_list, f)


# ==========================================
# 4. Logic & Integer Simplification
# ==========================================


def apply_dot_operator(expr):
    expr = expr.doit()
    for name, (func, func_dot, func_ddot) in dot_map.items():
        dt_X = func_dot - A * sp.diff(func, r)
        dt_dt_X = (
            func_ddot
            - 2 * A * sp.diff(func_dot, r)
            - A_dot * sp.diff(func, r)
            + A * sp.diff(A, r) * sp.diff(func, r)
            + (A**2) * sp.diff(func, r, r)
        )
        dt_dr_X = sp.diff(func_dot, r) - sp.diff(A, r) * sp.diff(func, r) - A * sp.diff(func, r, r)
        dt_dz_X = sp.diff(func_dot, z) - sp.diff(A, z) * sp.diff(func, r) - A * sp.diff(func, r, z)

        expr = expr.replace(sp.Derivative(func, t, t), dt_dt_X)
        expr = expr.replace(sp.Derivative(func, t, r), dt_dr_X)
        expr = expr.replace(sp.Derivative(func, r, t), dt_dr_X)
        expr = expr.replace(sp.Derivative(func, t, z), dt_dz_X)
        expr = expr.replace(sp.Derivative(func, z, t), dt_dz_X)
        expr = expr.replace(sp.Derivative(func, t), dt_X)
    return sp.simplify(expr)


def clean_equation(expr):
    """
    Cleans up the equation:
    1. Removes floats/fractions.
    2. Factors out non-zero terms (Exponentials, Sigma).
    3. Flips the sign if the leading term is negative.
    """
    # 1. Force conversion of floats to rationals
    expr = sp.nsimplify(expr, tolerance=1e-10, rational=True)

    # 2. Combine fractions
    expr = sp.together(expr)

    # 3. Extract Numerator
    expr, denom = sp.fraction(expr)

    # 4. Use primitive() to remove numeric content (GCD)
    # This removes the integer factors, but signs can still be tricky
    content, expr = sp.primitive(expr)

    # 5. Factorize to remove symbolic non-zero terms
    factored = sp.factor(expr)

    if factored.is_Mul:
        args = list(factored.args)
        new_args = []
        for term in args:
            if term.is_Number:
                continue  # Skip numbers (content handled by primitive)

            # Skip Exponentials and Sigma
            if isinstance(term, sp.exp) or (term.is_Pow and term.base == sp.E):
                continue
            if term == Sigma or (term.is_Pow and term.base == Sigma):
                continue

            new_args.append(term)

        if new_args:
            expr = sp.Mul(*new_args)
        else:
            expr = sp.Integer(1)

    # --- 6. The Fix: Flip overall sign if leading term is negative ---
    # We look at the ordered terms of the expression.
    # If the first term has a negative coefficient, we negate the whole expression.
    # e.g., -A - B = 0  --->  A + B = 0

    # Expand first to ensure we see separate terms (A+B) and not factored -(A+B)
    expr_expanded = sp.expand(expr)
    terms = expr_expanded.as_ordered_terms()

    if terms:
        first_term = terms[0]
        # Get the numerical coefficient of the first term
        coeff, _ = first_term.as_coeff_Mul()
        if coeff < 0:
            expr = -expr

    return expr


def display_ordered_equation(eqn, target_func, var=r, label=""):
    """
    Extracts coefficients for target_func and its derivatives,
    then constructs a LaTeX string manually to enforce the visual order:

    (Coeff) F'' + (Coeff) F' + (Coeff) F = RHS
    """

    # 1. Expand to ensure coefficients are separable
    expr = sp.expand(eqn)

    # 2. Define derivatives
    f_ddot = sp.Derivative(target_func, var, var)
    f_dot = sp.Derivative(target_func, var)

    # 3. Extract Coefficients
    coeff_ddot = sp.simplify(expr.coeff(f_ddot))
    remaining = expr - coeff_ddot * f_ddot

    coeff_dot = sp.simplify(remaining.coeff(f_dot))
    remaining = remaining - coeff_dot * f_dot

    coeff_func = sp.simplify(remaining.coeff(target_func))

    # 4. Extract RHS (Source Terms)
    # Since Expr = 0, RHS = -Remainder
    rhs = sp.expand(sp.simplify(-(remaining - coeff_func * target_func)))

    # =========================================================
    # 5. Manual LaTeX Construction (The Fix)
    # =========================================================

    parts = []

    # --- Helper to format term ---
    def add_term(coeff, deriv_latex):
        if coeff == 0:
            return

        # Get LaTeX for coefficient
        c_str = dot_latex(coeff)

        # Handle "1" and "-1" explicitly to avoid "1 \partial..."
        if coeff == 1:
            term = deriv_latex
        elif coeff == -1:
            term = "-" + deriv_latex
        else:
            # Wrap complex coefficients in parentheses if they are sums
            if coeff.is_Add:
                term = rf"\left( {c_str} \right) {deriv_latex}"
            else:
                term = rf"{c_str} {deriv_latex}"

        # Handle sign for joining
        if not parts:  # First term
            parts.append(term)
        else:
            # If the term starts with "-", just add it. Else add "+"
            if term.strip().startswith("-"):
                parts.append(term)
            else:
                parts.append("+" + term)

    # --- Build LHS in specific order ---

    # 1. Second Derivative
    add_term(coeff_ddot, dot_latex(f_ddot))

    # 2. First Derivative
    add_term(coeff_dot, dot_latex(f_dot))

    # 3. Function (0th Derivative)
    add_term(coeff_func, dot_latex(target_func))

    # If LHS is empty (0=RHS case), put "0"
    lhs_str = " ".join(parts) if parts else "0"

    # --- Build RHS ---
    rhs_str = dot_latex(rhs)

    if label:
        with open("equations.txt", "a", encoding="utf-8") as f:
            equation = f"$${lhs_str} = {rhs_str}$$\n"
            f.write(equation)

    # --- Final Display ---
    prefix = rf"{label}: \\\\ " if label else ""
    display(Math(rf"{prefix}{lhs_str} = {rhs_str}"))


# ==========================================
# 5. Display
# ==========================================
print("=" * 26 + "\nEinstein Tensor Components\n" + "=" * 26)

# reset equations file
open("equations.txt", "w").close()

for i in range(4):
    for j in range(i, 4):
        component = G_uv_list[i][j]

        if component != 0:
            transformed = apply_dot_operator(component)
            final_eq = clean_equation(transformed)

            label = rf"G_{{{i}{j}}}"

            latex_str = dot_latex(final_eq)
            display(Math(rf"{label}: \\\\ {latex_str} = 0"))
            with open("equations.txt", "a", encoding="utf-8") as f:
                equation = f"$${latex_str} = 0$$\n"
                f.write(equation)

            # display_ordered_equation(final_eq, F, label=label)
            # display_ordered_equation(final_eq, Sigma_dot, label=label)
            # display_ordered_equation(final_eq, B_dot, label=label)
            # display_ordered_equation(final_eq, A, label=label)
            print("-" * 180)


print("\n" + "=" * 50)
print("(2.8a) Extracting Σ Equation from G_22 (G_rr)")
print("=" * 50)

eq_Sigma = G_uv_list[2][2]
eq_Sigma = apply_dot_operator(eq_Sigma)
eq_Sigma = clean_equation(eq_Sigma)

display_ordered_equation(eq_Sigma, Sigma)

print("\n" + "=" * 50)
print("(2.8b) Extracting F Equation from G_12 (G_zr)")
print("=" * 50)

eq_F = G_uv_list[1][2]
eq_F = apply_dot_operator(eq_F)
eq_F = -clean_equation(eq_F)

display_ordered_equation(eq_F, F)

# validate 2.8b form
F_rr = sp.Derivative(F, r, r)
F_r = sp.Derivative(F, r)

coeff_Frr = eq_F.coeff(F_rr)
coeff_Fr = eq_F.coeff(F_r)
C_F = eq_F.coeff(F)

S_F_terms = eq_F - coeff_Frr * F_rr - coeff_Fr * F_r - C_F * F
S_F = -sp.simplify(S_F_terms)

print("Target Equation Structure: -Σ^2 * F_rr - Σ^2 * B_r * F_r + C_F * F = S_F")
print("-" * 60)

print("    Coeff of F_rr (Should be -Σ^2):")
display(Math(r"\qquad " + dot_latex(coeff_Frr)))

print("    Coeff of F_r (Should be -Σ^2 * B_r):")
display(Math(r"\qquad " + dot_latex(coeff_Fr)))

print("    Calculated C_F [B, Σ]:")
display(Math(r"\qquad " + dot_latex(C_F)))

print("    Calculated S_F [Σ, B]:")
display(Math(r"\qquad " + dot_latex(S_F)))

print("\n" + "=" * 50)
print("(2.8c) Extracting Σ_dot Equation from G_02 (G_tr) [after simplifying with G_22]")
print("=" * 50)

eq_Sigma_dot = G_uv_list[0][2]

eq_Sigma_dot = apply_dot_operator(eq_Sigma_dot)
eq_Sigma_dot = clean_equation(eq_Sigma_dot) / 2

d2_sigma_dr2 = sp.solve(eq_Sigma, Sigma.diff(r, r))[0]
eq_Sigma_dot = eq_Sigma_dot.subs(Sigma.diff(r, r), d2_sigma_dr2)

display_ordered_equation(eq_Sigma_dot, Sigma_dot)

# validate 2.8c form
Sigma_dot_r = sp.Derivative(Sigma_dot, r)

coeff_Sigma_dot_r = eq_Sigma_dot.coeff(Sigma_dot_r)
coeff_Sigma_dot = eq_Sigma_dot.coeff(Sigma_dot)

S_Sigma_dot_terms = eq_Sigma_dot - coeff_Sigma_dot_r * Sigma_dot_r - coeff_Sigma_dot * Sigma_dot
S_Sigma_dot = -sp.expand(sp.simplify(S_Sigma_dot_terms))

print("Target Equation Structure: 4Σ^3 * Σ_dot_r + 4Σ^2 * Σ_r * Σ_dot = S_Σ_dot")
print("-" * 60)

print("    Coeff of Σ_dot_r (Should be 4Σ^3):")
display(Math(r"\qquad " + dot_latex(coeff_Sigma_dot_r)))

print("    Coeff of Σ_dot (Should be 4Σ^2 * Σ_r):")
display(Math(r"\qquad " + dot_latex(coeff_Sigma_dot)))

print("    Calculated S_Σ_dot [Σ, B, F]:")
display(Math(r"\qquad " + dot_latex(S_Sigma_dot)))


print("\n" + "=" * 50)
print("(2.8d) Extracting B_dot Equation from 1/2 (G_11 (G_zz) - Σ * G_33 (G_xx))")
print("=" * 50)

# must clean for division by e^B
eq_B_dot = (clean_equation(G_uv_list[1][1]) - Sigma * clean_equation(G_uv_list[3][3])) / 2
eq_B_dot = apply_dot_operator(eq_B_dot)
eq_B_dot = clean_equation(eq_B_dot)

display_ordered_equation(eq_B_dot, B_dot)

# validate 2.8c form
B_dot_r = sp.Derivative(B_dot, r)

coeff_B_dot_r = eq_B_dot.coeff(B_dot_r)
coeff_B_dot = eq_B_dot.coeff(B_dot)

S_B_dot_terms = eq_B_dot - coeff_B_dot_r * B_dot_r - coeff_B_dot * B_dot
S_B_dot = -sp.simplify(S_B_dot_terms)

print("Target Equation Structure: 4Σ^4 * B_dot_r + 4Σ^3 * Σ_r * B_dot = S_B_dot")
print("-" * 60)

print("    Coeff of B_dot_r (Should be 4Σ^4):")
display(Math(r"\qquad " + dot_latex(coeff_B_dot_r)))

print("    Coeff of B_dot (Should be 4Σ^3 * Σ_r):")
display(Math(r"\qquad " + dot_latex(coeff_B_dot)))

print("    Calculated S_B_dot [Σ, B, F, Σ_dot]:")
display(Math(r"\qquad " + dot_latex(S_B_dot)))

print("\n" + "=" * 50)
# could have used: 1/2 Σ * G_33 (G_xx)
print("(2.8e) Extracting A Equation from 1/2 G_11 (G_zz)")
print("=" * 50)

eq_A = G_uv_list[1][1]
eq_A = apply_dot_operator(eq_A)
eq_A = clean_equation(eq_A) / 2

display_ordered_equation(eq_A, A)

# validate 2.8e form
A_rr = sp.Derivative(A, r, r)

coeff_Arr = eq_A.coeff(A_rr)

S_A_terms = eq_A - coeff_Arr * A_rr
S_A = -sp.simplify(S_A_terms)

print("Target Equation Structure: 2Σ^4 * A_rr = S_A")
print("-" * 60)

print("    Coeff of A_rr (Should be 2Σ^4):")
display(Math(r"\qquad " + dot_latex(coeff_Arr)))

print("    Calculated S_A [Σ, B, F Σ_dot, B_dot]:")
display(Math(r"\qquad " + dot_latex(S_A)))
# %%
