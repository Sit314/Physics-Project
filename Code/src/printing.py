from sympy.printing.latex import LatexPrinter


class DotPrinter(LatexPrinter):
    # Define colors once as a class constant
    COLOR_MAP = {
        "A": {"base": "#FF6666", "dot": "#FFAAAA", "ddot": "#CC0000"},
        "B": {"base": "#44AAFF", "dot": "#88CCFF", "ddot": "#0066CC"},
        "F": {"base": "#32CD32", "dot": "#98FB98", "ddot": "#008800"},
        "Sigma": {"base": "#FFC125", "dot": "#FFE082", "ddot": "#FF8C00"},
    }

    def _print_Function(self, expr, exp=None):
        func_name = expr.func.__name__

        if func_name.endswith("_ddot"):
            base, deriv_type = func_name.replace("_ddot", ""), "ddot"
        elif func_name.endswith("_dot"):
            base, deriv_type = func_name.replace("_dot", ""), "dot"
        else:
            base, deriv_type = func_name, "base"

        base_latex = r"\Sigma" if base == "Sigma" else base

        if base in self.COLOR_MAP:
            colors = self.COLOR_MAP[base]

            if deriv_type == "ddot":
                symbol_str = rf"\ddot{{{base_latex}}}"
                color = colors["ddot"]
            elif deriv_type == "dot":
                symbol_str = rf"\dot{{{base_latex}}}"
                color = colors["dot"]
            else:
                symbol_str = base_latex
                color = colors["base"]

            colored_symbol = rf"{{\color{{{color}}} {symbol_str}}}"

            if exp:
                return rf"{colored_symbol}^{{{exp}}}"
            return colored_symbol

        return super()._print_Function(expr, exp)

    def _print_Derivative(self, expr):
        function = expr.args[0]
        func_str = self._print(function)
        indices_str = "".join([self._print(var) * count for var, count in expr.variable_count])
        return rf"\partial_{{{indices_str}}} {func_str}"


def dot_latex(expr):
    return DotPrinter().doprint(expr)
