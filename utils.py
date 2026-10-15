def contains_emoji(text):
    if not isinstance(text, str):
        return False

    return any(
        '\U0001F300' <= char <= '\U0001FAFF'
        or '\U00002600' <= char <= '\U000027BF'
        for char in text
    )