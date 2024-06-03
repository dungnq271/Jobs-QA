import re


def remove_redundant_whitespaces(text: str):
    """Replace redundant whitespaces by a single whitespace."""
    return re.sub(r"\s{2,}", " ", text)
