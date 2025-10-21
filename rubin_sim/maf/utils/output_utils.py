__all__ = ("name_sanitize",)


def name_sanitize(in_string):
    """
    Convert a string to a more file name (and web) friendly format.

    Parameters
    ----------
    in_string : `str`
        The input string to be sanitized.
        Typically these are combinations of metric names and metadata.

    Returns
    -------
    out_string : `str`
        The string after removal/replacement of non-friendly characters.
    """
    # Replace <, > and = signs.
    out_string = in_string.replace(">", "gt").replace("<", "lt").replace("=", "eq")
    # Remove single-spaces, strip '.'s and ','s
    out_string = out_string.replace(" ", "_").replace(".", "_").replace(",", "")
    # and remove / and \
    out_string = out_string.replace("/", "_").replace("\\", "_")
    # and remove parentheses
    out_string = out_string.replace("(", "").replace(")", "")
    # Remove ':' and ';"
    out_string = out_string.replace(":", "_").replace(";", "_")
    # Replace '%' and #
    out_string = out_string.replace("%", "_").replace("#", "_")
    # Remove '__'
    while "__" in out_string:
        out_string = out_string.replace("__", "_")
    return out_string
