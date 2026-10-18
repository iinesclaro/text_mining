# imports
import re
import unicodedata
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer

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
    flags = re.IGNORECASE)

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

    emoji_pattern = (
    r'[\U0001F300-\U0001FAFF'
    r'\u2600-\u26FF'
    r'\u2700-\u27BF'
    r'\u2300-\u23FF'
    r'\u2B00-\u2BFF'
    r'\u203C\u2049'
    r'\uFE0F]')
    
    return bool(re.search(emoji_pattern, text))


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

def contains_hashtag(text):
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

    pattern = re.compile(r'<[^>]+>')

    return bool(pattern.search(text))

def contains_commercial_symbols(text):
    
    if not isinstance(text, str):
            return False

    symbols = r'[®™©℠℗]'
    
    return bool(re.search(symbols, text))

def contains_italics(text):

    if not isinstance(text, str):
                return False

    return any('MATHEMATICAL' in unicodedata.name(char, '') and 'ITALIC' in unicodedata.name(char, '') for char in text)

def remove_emojis(text):

    emoji_pattern = (
    r'[\U0001F300-\U0001FAFF'
    r'\u2600-\u26FF'
    r'\u2700-\u27BF'
    r'\u2300-\u23FF'
    r'\u2B00-\u2BFF'
    r'\u203C\u2049'
    r'\uFE0F]')
    
    return re.sub(emoji_pattern, "", text)

def remove_handles(text):
    return re.sub(r'(?<!\w)@[A-Za-z0-9_]+', "" , text)

def remove_hashtags(text):
    return re.sub(r'(?<!\w)#[A-Za-z0-9_]+', "" ,text)

def remove_urls(text):
    return re.sub(r'(https?://|www\.)\S+', " " , text, flags = re.IGNORECASE)

def remove_htmls(text):
    return re.sub(r'<[^>]+>', "" , text)

def remove_commercial_symbols(text):

    symbols = r'[®™©℠℗]'

    return re.sub(symbols, '', text)

def normalize_newlines(text):
    return re.sub('\n', " " , text)

def normalize_apostrophes(text):

    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('‛', "'")
    text = text.replace('‚', "'")
    text = text.replace('´', "'")

    return text

def normalize_whitespaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_structural_content(text):

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
        flags = re.IGNORECASE)

    return re.sub(pattern, '', text)

def remove_punctuation(text):
    
    text = re.sub(r'<[^>]*(?:>|$)', ' ', text)

    text = re.sub(r'\b\d+x\b', ' ', text, flags=re.IGNORECASE)
    
    text = re.sub(r'[-\u2010-\u2015\u2212]+', ' ', text)

    punctuation = r"""[!"$%&()*+,./:;<=>?\[\\\]^_`{|}~\-]"""

    text = re.sub(punctuation, ' ', text)

    return text

def remove_italics(text):
    return unicodedata.normalize('NFKC', text)

stop_words = set(stopwords.words('english'))
stop_words_with_negations = stop_words - {'no', 'nor', 'not'}

def stopword_remover(tokens):
    clean_tokens = []

    for token in tokens:
        if token.casefold() not in stop_words:
            clean_tokens.append(token)
    return clean_tokens

def tokenizer(text):
    return word_tokenize(str(text))

lemmatizer = WordNetLemmatizer()
sentence_tokenizer = PunktSentenceTokenizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # default is noun

def normalize_lyric_expressions(text):
    
    text = re.sub(r"\bgon\s+na\b", "gonna", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwan\s+na\b", "wanna", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgot\s+ta\b", "gotta", text, flags=re.IGNORECASE)

    return text

def clean_lyrics(text, 
                keep_hashtags = True, 
                keep_emojis = True, 
                keep_structural_content = True, 
                keep_punctuation = True, 
                lower_case = False, 
                lemmatization = False, 
                remove_stopwords = False, 
                tokenized_output = False, 
                sentence_output = True, 
                keep_negations = False):

    """Clean lyrics and return tokens, a cleaned string, or sentence strings.

    When ``tokenized_output`` is true, it takes precedence over
    ``sentence_output`` and the result is a list of tokens.
    """

    if not isinstance(text, str):
        return [] if tokenized_output else ''
    
    text = text.replace("\u0435", "e")
    text = normalize_apostrophes(text)
    text = normalize_lyric_expressions(text)
    text = remove_commercial_symbols(text)
    text = remove_urls(text)
    text = remove_handles(text)
    text = remove_htmls(text)
    text = normalize_newlines(text)
    text = remove_italics(text)

    if keep_hashtags == False:
        text = remove_hashtags(text)
    if keep_emojis == False:
        text = remove_emojis(text)
    if keep_structural_content == False:
        text = remove_structural_content(text)
    if keep_punctuation == False:
        text = remove_punctuation(text)
    if lower_case:
        text = text.lower()
    text = normalize_whitespaces(text)
    text = text.strip()

    tokens = tokenizer(text)
    
    corrected_tokens = []
    i = 0

    while i < len(tokens):

        # Join split expressions: wanna, gonna, gotta
        if i + 1 < len(tokens) and (
            tokens[i].casefold(), tokens[i + 1].casefold()
        ) in [
            ("wan", "na"),
            ("gon", "na"),
            ("got", "ta")
        ]:
            corrected_tokens.append(
                tokens[i].casefold() + tokens[i + 1].casefold()
            )
            i += 2

        # Join contractions such as I'm, you're, don't, and ain't
        elif i + 1 < len(tokens) and tokens[i + 1] in [
            "'m", "'s", "'re", "'ve", "'ll", "'d", "n't"
        ]:
            corrected_tokens.append(tokens[i] + tokens[i + 1])
            i += 2

        # Join words ending in an apostrophe, such as stickin'
        elif (
            i + 1 < len(tokens)
            and tokens[i + 1] in ["'", "’"]
        ):
            corrected_tokens.append(tokens[i] + "'")
            i += 2

        else:
            corrected_tokens.append(tokens[i])
            i += 1

    tokens = corrected_tokens

    if remove_stopwords:
        if keep_negations:
            tokens = [
            token for token in tokens
            if token.casefold() not in stop_words_with_negations]
        else:
            tokens = stopword_remover(tokens)

    if lemmatization:
        tagged_tokens = pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]

    if tokenized_output:
        return tokens

    text = " ".join(tokens)
    if sentence_output:
        text = sentence_tokenizer.tokenize(str(text))

    return text