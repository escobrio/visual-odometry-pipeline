import numpy as np

def format_info(info: dict, header: str | None = None) -> str:
    """Recursively format nested dictionaries into a clean string, with optional header.
        Formatting inoprints.
    input:
        info: dict
            Nested dictionary to format.
        header: str | None
            Optional header to add at the top of the formatted string.
    output:
        str: Formatted string representation of the nested dictionary.
    """
    lines: list[str] = []

    # Add header only once, at the very top
    if header is not None:
        lines.append(header)
        lines.append("-" * len(header))

    def _recurse(d: dict, indent: int = 0) -> None:
        prefix = "    " * indent  # 4-space indentation

        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                _recurse(value, indent + 1)
            else:
                lines.append(f"{prefix}{key}: {value}")

    _recurse(info, indent=0)

    # add two empty lines at the end for better separation
    lines.append("")
    lines.append("")

    return "\n".join(lines)
