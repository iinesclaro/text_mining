# imports
import re

# 1. contains_lyrics

def contains_lyrics(text):
    '''
    Checks if the text string contains at least one common lyric-related keyword.

    Parameters
    ----------
    text: str
        The text to inspect.
    
    Returns
    ----------
    bool
        True if at least one keyword is found, False otherwise.
    '''
    if not isinstance(text, str): # this is an insurance check that checks if text is of string-type
        return False

    pattern = re.compile(
    r'(?:'
    r'[\[(](?:'
    r'intro|'
    r'verse(?:\s+\d+)?|'
    r'chorus|'
    r'hook|'
    r'pre-chorus|'
    r'post-chorus|'
    r'bridge|'
    r'outro|'
    r'breakdown|'
    r'interlude|'
    r'refrain'
    r')(?:\s*:\s*[^)\]]*)?\s*[\])]'
    r'|'
    r'\b(?:verse(?:\s+\d+)?|chorus|hook|intro|outro|bridge)'
    r'\s*:'
    r')',
    flags = re.IGNORECASE
)

    # the IGNORECASE flags allows for uppercase and lowercase matches

    return bool(pattern.search(text))

# 2. contains_repeated_lines

def contains_repeated_lines(text, min_line_length = 15, min_repetitions = 2):
    if not isinstance(text, str):
        return False

    lines = [
        re.sub(r'\s+', ' ', line.strip().lower())
        for line in text.split("\n")
    ]

    lines = [
        line for line in lines
        if len(line) >= min_line_length
    ]

    for i in range(len(lines)):
        count = 1

        for j in range(i + 1, len(lines)):
            if lines[i] == lines[j]:
                count += 1

                if count >= min_repetitions:
                    return True

    return False

# 2. contains_emoji

def contains_emoji(text):
    '''
    Checks if a text string contains at least one characther within the common Unicode emoji ranges.

    Parameters
    ----------
    text: str
        The text to inspect.

    Returns
    ----------
    bool
        True if at least one of the characters is within the ranges, False otherwise.
    '''
    if not isinstance(text, str): 
        return False

    return bool(re.search(r'[\u2600-\u27FF]', text))


# 3. contains_handle

def contains_handle(text):
    '''
    Checks if a text string contains at least one handle (@).

    Parameters
    ----------
    text: str
        The text to inspect.
    
    Returns
    --------
    bool
        True if a @ is found, False otherwise.
    '''
    if not isinstance(text, str):
            return False
    
    return bool(re.search(r'(?<!\w)@[A-Za-z0-9_]+', text))
    # this regex pattern searches for @ that are not preceeded by words
    # example: matches @username but does not match 20241111@novaims.unl.pt, since there is a 1 before it

# 4. contains_hastag

def contains_hastag(text):
    '''
    Checks if a text string contains at least one hastag (#).

    Parameters
    ----------
    text: str
        The text to inspect.
    Returns
    --------
    bool
        True if a # is found, False otherwise.

    '''
    if not isinstance(text, str):
        return False

    return bool(re.search(r'(?<!\w)#[A-Za-z0-9_]+', text))
    # this regex pattern searches for # that are not preceeded by words.
    # example: matches #textmining but does not match example#tag or text#mining@email.

# 5. contains_url

def contains_url(text):
    '''
    Checks if a text string contains at least one URL.

    Parameters
    ----------
    text : str
        The text to inspect.

    Returns
    -------
    bool
        True if a URL is found, False otherwise.
    '''
    if not isinstance(text, str):
        return False

    return bool(re.search(r'(https?://|www\.)\S+', text, flags = re.IGNORECASE))
    # this pattern matches websites starting with http://, https://, or www. followed by a non-whitespace character.
    # example: matches http://example.org or https://example.pt but does not match google.com.

# 6. contains_newline

def contains_newline(text):
    '''
    Checks if a text string contains at least one newline character.

    Parameters
    ----------
    text : str
        The text to inspect.

    Returns
    -------
    bool
        True if a newline is found, False otherwise.
    '''
    if not isinstance(text, str):
        return False

    return '\n' in text

# 7. contains_html

def contains_html(text):
    '''
    Checks if the text string contains HTML tags.

    Parameters
    ----------
    text : str
        The text to inspect.

    Returns
    -------
    bool
        True if HTML tags are found, False otherwise.
    '''
    if not isinstance(text, str):
        return False

    regex = re.compile(r'<[^>]+>')

    return bool(regex.search(text))
