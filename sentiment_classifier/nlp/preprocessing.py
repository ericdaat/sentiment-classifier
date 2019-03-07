import re


def clean_text(text):
    text = text.strip()
    # remove html
    text = re.sub(string=text, pattern=r"<[^>]*>", repl="")
    # add spaces between special characters
    text = re.sub(string=text,
                  pattern=r"([$&+,:;=?@#|\"<>.^*()%!-])",
                  repl=r" \1 ")
    text = text.strip()

    return text
