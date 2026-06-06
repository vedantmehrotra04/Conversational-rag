def to_list(text):
    return [
        item.strip("- ").strip()
        for item in text.split("\n")
        if item.strip()
    ]