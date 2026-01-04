import sympy as sp
from IPython.display import display, Math
from src.printing import dot_latex
from src.config import r


def display_ordered_equation(eqn, target_func, var=r, label=""):
    """
    Manually constructs LaTeX to enforce order: F'' + F' + F = RHS
    """
    expr = sp.expand(eqn)
    f_ddot = sp.Derivative(target_func, var, var)
    f_dot = sp.Derivative(target_func, var)

    # Extract Coefficients
    coeff_ddot = sp.simplify(expr.coeff(f_ddot))
    remaining = expr - coeff_ddot * f_ddot

    coeff_dot = sp.simplify(remaining.coeff(f_dot))
    remaining = remaining - coeff_dot * f_dot

    coeff_func = sp.simplify(remaining.coeff(target_func))

    # RHS is the negative of the remainder (since Expr = 0)
    rhs = sp.expand(sp.simplify(-(remaining - coeff_func * target_func)))

    # Construction
    parts = []

    def add_term(coeff, deriv_latex):
        if coeff == 0:
            return
        c_str = dot_latex(coeff)

        if coeff == 1:
            term = deriv_latex
        elif coeff == -1:
            term = "-" + deriv_latex
        elif coeff.is_Add:
            term = rf"\left( {c_str} \right) {deriv_latex}"
        else:
            term = rf"{c_str} {deriv_latex}"

        if not parts:
            parts.append(term)
        elif term.strip().startswith("-"):
            parts.append(term)
        else:
            parts.append("+" + term)

    add_term(coeff_ddot, dot_latex(f_ddot))
    add_term(coeff_dot, dot_latex(f_dot))
    add_term(coeff_func, dot_latex(target_func))

    lhs_str = " ".join(parts) if parts else "0"
    rhs_str = dot_latex(rhs)

    # Write to file
    if label:
        with open("equations.txt", "a", encoding="utf-8") as f:
            f.write(f"$${lhs_str} = {rhs_str}$$\n")

    prefix = rf"{label}: \\\\ " if label else ""
    display(Math(rf"{prefix}{lhs_str} = {rhs_str}"))
