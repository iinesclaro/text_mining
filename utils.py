# IMPORTS

import re
import unicodedata
import numpy as np
import itertools

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from joblib import Parallel, delayed

# models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# vectorizers

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from Logger import CheckTimer
# INITIALIZATIONS

sentence_tokenizer = PunktSentenceTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# PATTERNS

emoji_pattern = (
    r'[\U0001F300-\U0001FAFF'
    r'\u2600-\u26FF'
    r'\u2700-\u27BF'
    r'\u2300-\u23FF'
    r'\u2B00-\u2BFF'
    r'\u203C\u2049'
    r'\uFE0F]')

handle_pattern = (r'(?<!\w)@[A-Za-z0-9_]+')

hashtag_pattern = (r'(?<!\w)#[A-Za-z0-9_]+')

url_pattern = (r'(?:https?://|www\.)\S+')

commercial_symbols_pattern = (r'[®™©℠℗]')

punctuation_pattern = (r'[!\"$%&()*+,./:;<=>?\[\\\]^_`{|}~\-]')

# STOPWORDS

## APPROACH 1
stop_words = set(stopwords.words('english'))

## APPROACH 2
pronouns = {
    'i', 'me', 'my', 'myself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'they', 'them', 'their', 'theirs', 'themselves',
    'it', 'its', 'itself'
}

negations = {
    'no', 'nor', 'not',
    'ain', 'aren', "aren't",
    'couldn', "couldn't",
    'didn', "didn't",
    'doesn', "doesn't",
    'don', "don't",
    'hadn', "hadn't",
    'hasn', "hasn't",
    'haven', "haven't",
    'isn', "isn't",
    'mightn', "mightn't",
    'mustn', "mustn't",
    'needn', "needn't",
    'shan', "shan't",
    'shouldn', "shouldn't",
    'wasn', "wasn't",
    'weren', "weren't",
    'won', "won't",
    'wouldn', "wouldn't"
}

auxiliary_verbs = {
    'am', 'are', 'be', 'been', 'being',
    'can', 'could',
    'did', 'do', 'does', 'doing',
    'had', 'has', 'have', 'having',
    'is', 'was', 'were',
    'will', 'would', 'should',
    'might', 'must', 'need'
}

question_words = {
    'how', 'what', 'when', 'where',
    'which', 'who', 'whom', 'why'
}

other_words = {
    'all', 'any', 'because', 'but',
    'down', 'if', 'into', 'just',
    'more', 'most', 'never', 'now',
    'only', 'or', 'own', 'so',
    'some', 'than', 'that', 'then',
    'there', 'these', 'this', 'those',
    'through', 'too', 'up', 'very',
    'with'
}

words_to_keep = (
    pronouns
    | negations
    | auxiliary_verbs
    | question_words
    | other_words
)

custom_stopwords = set(stopwords.words('english')) - words_to_keep

# STOPWORD REMOVER

def stopword_remover(tokens, custom = "none"):
    ''' 
    takes as input tokens and the custom parameter

    if custom is none no stop words will be removed
    if custom is standard the standard stop words will be removed
    if custom is custom the custom stop words defined above will be removed 

    by default custom is equal to none
    '''
    clean_tokens = []


    if custom  == 'custom':
        words = custom_stopwords
    elif custom == 'standard':
        words = stop_words

    elif custom == 'none':
        return tokens

    else:
        return tokens

    for token in tokens:
        if token.casefold() not in words:
            clean_tokens.append(token)
    return clean_tokens

# FUNCTIONS: CONTAINS

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
    
    return bool(re.search(emoji_pattern, text))


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
    
    return bool(re.search(handle_pattern, text))
    # this regex pattern searches for @ that are not preceeded by words
    # example: matches @username but does not match 20241111@novaims.unl.pt, since there is a 1 before it


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

    return bool(re.search(hashtag_pattern, text))
    # this regex pattern searches for # that are not preceeded by words.
    # example: matches #textmining but does not match example#tag or text#mining@email.


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

    return bool(re.search(url_pattern, text, flags=re.IGNORECASE))
    # this pattern matches websites starting with http://, https://, or www. followed by a non-whitespace character.
    # example: matches http://example.org or https://example.pt but does not match google.com.

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

def contains_angle_bracket_markers(text):
    if not isinstance(text, str):
        return False

    # repetition markers like <2x, <3x
    if re.search(r'(?im)^\s*<\s*\d+\s*x\s*$', text):
        return True

    # > or < at the start of a line
    return bool(re.search(r'(?m)^\s*[<>]\s*', text))

def contains_commercial_symbols(text):
    
    if not isinstance(text, str):
            return False
    
    return bool(re.search(commercial_symbols_pattern, text))

def contains_italics(text):

    if not isinstance(text, str):
                return False

    return any('MATHEMATICAL' in unicodedata.name(char, '') and 'ITALIC' in unicodedata.name(char, '') for char in text)

# FUNCTIONS: REMOVE

def remove_emojis(text):
    return re.sub(emoji_pattern, "", text)

def remove_handles(text):
    return re.sub(handle_pattern, "" , text)

def remove_hashtags(text, keep_words = True):
    if keep_words:
        return re.sub(r'(?<!\w)#(?=[A-Za-z0-9_])', '', text)
    
    return re.sub(hashtag_pattern, ' ', text)

def remove_urls(text):
    return re.sub(url_pattern, " " , text, flags = re.IGNORECASE)

def remove_angle_bracket_markers(text):
    if not isinstance(text, str):
        return text

    # removes repetition markers like <2x, <3x
    text = re.sub(r'(?im)^\s*<\s*\d+\s*x\s*$', '', text)

    # removes > or < at the start of a line
    text = re.sub(r'(?m)^\s*[<>]\s*', '', text)

    return text

def remove_commercial_symbols(text):
    return re.sub(commercial_symbols_pattern, '', text)

def remove_structural_content(text):

    structural_content = re.compile(
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

    return re.sub(structural_content, '', text)

def remove_punctuation(text):

    # repetitions like 3x, 2x
    text = re.sub(r'\b\d+x\b', ' ', text, flags=re.IGNORECASE)

    # dash-characters
    text = re.sub(r'[-\u2010-\u2015\u2212]+', ' ', text)

    text = re.sub(punctuation_pattern, ' ', text)

    return text

# FUNCTIONS: NORMALIZE

def normalize_newlines(text):
    return re.sub('\n', " " , text)

def normalize_apostrophes(text):
    for apostrophe in ["’", "‘", "‛", "‚", "´", "`", "′", "ʼ"]:
        text = text.replace(apostrophe, "'")

    return text

def normalize_whitespaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text):
    return unicodedata.normalize('NFKC', text).encode('ascii', 'ignore').decode('utf-8')

def normalize_lyric_expressions(text):
    
    text = re.sub(r"\bgon\s+na\b", "gonna", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwan\s+na\b", "wanna", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgot\s+ta\b", "gotta", text, flags=re.IGNORECASE)

    return text

# TOKENIZER
def tokenizer(text):
    return word_tokenize(str(text))

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

# MAIN PIPELINE FOR CLEANING LYRICS
def clean_lyrics(text, 
                   keep_hashtags = True,
                   keep_hashtags_words = True, 
                   keep_emojis = True, 
                   keep_structural_content = True, 
                   keep_punctuation = True, 
                   lower_case = False, 
                   lemmatization = False,
                   stemming = False, 
                   remove_stopwords = False,
                   custom_stopwords = True, 
                   tokenized_output = False, 
                   sentence_output = False):

    """Clean lyrics while applying optional removals before Unicode normalization.

    Returns tokens when ``tokenized_output`` is true; otherwise returns either
    sentence strings or one cleaned string, according to ``sentence_output``.
    """
    if not isinstance(text, str):
        return [] if tokenized_output else ''

    text = text.replace("\u0435", "e")
    text = normalize_apostrophes(text)
    text = normalize_lyric_expressions(text)
    text = remove_commercial_symbols(text)
    text = remove_urls(text)
    text = remove_handles(text)
    text = remove_angle_bracket_markers(text)
    text = normalize_newlines(text)

    if not keep_hashtags:
            text = remove_hashtags(text, keep_words = keep_hashtags_words)

    if not keep_emojis:
        text = remove_emojis(text)

    if not keep_structural_content:
        text = remove_structural_content(text)

    text = normalize_text(text)

    if lower_case:
        text = text.lower()

    if not keep_punctuation:
        text = remove_punctuation(text)

    text = normalize_whitespaces(text)

    tokens = tokenizer(text)
    corrected_tokens = []
    i = 0

    while i < len(tokens):
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
        elif i + 1 < len(tokens) and tokens[i + 1].lower() in [
            "'m", "'s", "'re", "'ve", "'ll", "'d", "n't"
        ]:
            corrected_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        elif i + 1 < len(tokens) and tokens[i + 1] in ["'", "’"]:
            corrected_tokens.append(tokens[i] + "'")
            i += 2
        else:
            corrected_tokens.append(tokens[i])
            i += 1

    tokens = [
        token for token in corrected_tokens
        if token.casefold().strip("'’‘‛‚´`′ʼ") not in {
            's', 't', 'm', 're', 'll', 've', 'd', 'em', 'g'
        }
    ]

    if remove_stopwords:
        tokens = stopword_remover(tokens, custom = custom_stopwords)

    if lemmatization:
        tagged_tokens = pos_tag(tokens)
        tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in tagged_tokens]

    elif stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    if tokenized_output:
        return tokens

    text = " ".join(tokens)
    if sentence_output:
        text = sentence_tokenizer.tokenize(text)

    return text


# Function to infer vectors for documents
def infer_vectors(model, docs):
    vectors = [model.infer_vector(doc) for doc in docs]
    return vectors


# -------------------------------------------------------------------------

def tokenize_with_stopwords(sw_mode, X):
    tokens_list = []
    for doc in X:
        if isinstance(doc, str):
            tokens_list.append(stopword_remover(doc.split(), sw_mode))
        else:
            # If it's a NaN/float/etc., append an empty list so it doesn't crash
            tokens_list.append([])

    return tokens_list

def to_dict_list(tokenized_list):
    return [{w: doc.count(w) for w in set(doc)} for doc in tokenized_list]

def list_to_raw_text(tokenized_list):
    return [" ".join(doc) for doc in tokenized_list]

def w2v_output(tokens_list, model, vector_size):
    vectors = []
    for words in tokens_list:
        word_vecs = [model.wv[w] for w in words if w in model.wv]
        vectors.append(np.mean(word_vecs, axis=0) if word_vecs else np.zeros(vector_size))
    return np.array(vectors)
    
# Function to apply Feature Extractor

def feature_extractor(vec_name, tokenized_train, tokenized_val, pretrained_model=None):


    if vec_name == 'DictVectorizer':
        train_dict =to_dict_list(tokenized_train)
        val_dict = to_dict_list(tokenized_val)

        dv = DictVectorizer(sparse=True)
        return dv.fit_transform(train_dict), dv.transform(val_dict)
    
    elif vec_name == 'CountVectorizer' or vec_name == 'TfidfVectorizer':

        raw_train = list_to_raw_text(tokenized_train)
        raw_val = list_to_raw_text(tokenized_val)

        if vec_name == 'CountVectorizer':
            model = CountVectorizer(max_features=10000, min_df=5)
        elif vec_name == 'TfidfVectorizer':
            model = TfidfVectorizer(max_features=10000, min_df=5)

        return model.fit_transform(raw_train), model.transform(raw_val)

    elif vec_name == 'Word2Vec':
        vector_size = 300

        model = Word2Vec(sentences=tokenized_train, vector_size=vector_size, window=5, min_count=1, workers=4)

        return w2v_output(tokenized_train, model, vector_size), w2v_output(tokenized_val, model, vector_size)

    elif vec_name == 'Doc2Vec':
        vector_size = 300

        tagged_tr = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_train)]
        model = Doc2Vec(vector_size=vector_size, window=5, min_count=1, epochs=5, workers=4)
        model.build_vocab(tagged_tr)
        model.train(tagged_tr, total_examples=model.corpus_count, epochs=model.epochs)
        
        return np.array([model.infer_vector(doc) for doc in tokenized_train]), np.array([model.infer_vector(doc) for doc in tokenized_val])



def choose_model(model_name):
    if model_name == 'LogisticRegression':
        return LogisticRegression(max_iter=2000, random_state=42)

    elif model_name == 'RandomForest':
        return RandomForestClassifier(random_state=42)

    elif model_name == 'NeuralNetwork':
        return MLPClassifier(max_iter=500, random_state=42, early_stopping=True)


def grid_search(vec_options, model_options, sw_options, X_train, y_train, 
                        n_splits = 5,  minimize=False):
    
    combinations = list(itertools.product(vec_options, model_options, sw_options))

    global_best_train_fitness = float('inf') if minimize else float('-inf')
    global_best_val_fitness = float('inf') if minimize else float('-inf')
    best_combo = None

    print(f'Grid Search starting\n{len(combinations)} combinations being tested')
    print('-' * 60)

    print("Pre-tokenizing data for each stopwords mode...")
    pretokenized_dict = {}

    for sw_option in sw_options:
        pretokenized_dict[sw_option] = tokenize_with_stopwords(sw_mode=sw_option, X=X_train)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for idx,combo in enumerate(combinations):
        
        vec_name, model_name, sw_option = combo
        print(f'Testing combo # {idx+1}')
    
        tokenized_data = pretokenized_dict[sw_option]
        combo_train_fit = []
        combo_val_fit = []

        for train_index, val_index in cv.split(X_train, y_train):
            tokenized_tr = [tokenized_data[i] for i in train_index]
            tokenized_va = [tokenized_data[i] for i in val_index]
        
            y_tr = y_train.iloc[train_index]
            y_va = y_train.iloc[val_index]

            # Extract features (Word2Vec/Doc2Vec train strictly on tokenized_tr per fold)
            Xtrain_ft, Xval_ft = feature_extractor(vec_name, tokenized_tr, tokenized_va, sw_option)

            base_model = choose_model(model_name)
            base_model.fit(Xtrain_ft, y_tr)

            train_predictions = base_model.predict(Xtrain_ft)
            val_predictions = base_model.predict(Xval_ft)

            train_score = f1_score(y_tr, train_predictions, average='weighted', zero_division=0)
            val_score = f1_score(y_va, val_predictions, average='weighted', zero_division=0)

            combo_train_fit.append(train_score)
            combo_val_fit.append(val_score)

    avg_train_fitness = np.mean(combo_train_fit)
    avg_val_fitness = np.mean(combo_val_fit)

    is_better = (avg_val_fitness < global_best_val_fitness) if minimize else (avg_val_fitness > global_best_val_fitness)
    if is_better:
        global_best_val_fitness = avg_val_fitness
        global_best_train_fitness = avg_train_fitness
        best_combo = combo

    print(f'\nBest combo: {best_combo}\nTrain fitness: {global_best_train_fitness}\nValidation fitness: {global_best_val_fitness}')
    return best_combo, global_best_train_fitness, global_best_val_fitness


def fold_score_calculator(y_pred, y_test):
    
    # Compute classification scores (accuracy, precision, recall, F1, AUC) for the fold.
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    return (acc, prec, recall, f1)







