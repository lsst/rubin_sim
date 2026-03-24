__all__ = ("get_filter_colors", "get_filter_symbols", "get_filter_linestyles", "bright_filter_colors")


def get_filter_colors(background="white"):
    """Colors for
    Defined at https://rtn-045.lsst.io/

    Parameters
    ----------
    background : `str`
        Color of the plot background. Default `white`. Other
        option `black`.
    """
    result = {"u": "#1600ea", "g": "#31de1f", "r": "#b52626", "i": "#370201", "z": "#ba52ff", "y": "#61a2b3"}
    if background == "black":
        result = {
            "u": "#3eb7ff",
            "g": "#30c39f",
            "r": "#ff7e00",
            "i": "#2af5ff",
            "z": "#a7f9c1",
            "y": "#fdc900",
        }
    return result


def get_filter_symbols():
    """Defined at https://rtn-045.lsst.io/"""
    return {"u": "o", "g": "^", "r": "v", "i": "s", "z": "*", "y": "p"}


def get_filter_linestyles():
    """Defined at https://rtn-045.lsst.io/"""
    return {"u": "--", "g": (0, (3, 1, 1, 1)), "r": "-.", "i": "-", "z": (0, (3, 1, 1, 1, 1, 1)), "y": ":"}


def bright_filter_colors():
    """For when one wants to ignore the project-approved
    colors and use more intuitive colors
    """
    return {"u": "purple", "g": "blue", "r": "green", "i": "cyan", "z": "orange", "y": "red"}
