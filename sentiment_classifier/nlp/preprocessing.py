""" Module for text pre-processing
"""

import re


def clean_text(text):
    """ Function to clean a string.
    This function does the following:

    - Remove HTML tags
    - Surround punctuation and special characters by spaces
    - Remove extra spaces

    Args:
        text (str): text to clean

    Returns:
        text (str): the cleaned text

    """
    # remove html
    text = re.sub(string=text, pattern=r"<[^>]*>", repl="")
    # add spaces between special characters
    text = re.sub(string=text,
                  pattern=r"([$&+,:;=?@#|\"<>.^*()%!-])",
                  repl=r" \1 ")
    text = text.strip()

    return text
