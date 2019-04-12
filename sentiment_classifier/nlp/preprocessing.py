""" Module for text pre-processing

We provide a basic test preprocessing function, that does the following tasks:

 - Removes HTML
 - Surround punctuation and special characters by spaces

This function can be passed to a Reader instance when loading the dataset.

Note: we did not lowercase the sentence, or removed the special characters \
    on purpose. We think this information can make a difference in \
    classifying sentiments. We are also using Word Embeddings, \
    and the embeddings are different on lowercase vs uppercase words.
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
