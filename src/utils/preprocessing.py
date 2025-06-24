import re

def preprocess_content_text(text):
    """
    Clean content text by removing semicolons and collapsing whitespace,
    while preserving meaningful content.
    """
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[;]', ' ', text.lower())  # Only remove semicolons
    return ' '.join(text.split())  # Collapse whitespace