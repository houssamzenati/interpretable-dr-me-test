import math
import os


def tex_float(value, digits=3):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def write_booktabs_table(path, align, body_lines, caption, label, position="t"):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    lines = [
        f"\\begin{{table}}[{position}]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
    ]
    lines.extend(body_lines)
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))
