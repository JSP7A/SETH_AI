import re
from seth_core.chat_engine import call_llm  

def extract_keywords(text, max_keywords=8):
    """
    Simple keyword extractor using regex filtering for words
    with at least 4 letters, ignoring a list of common words.
    """
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    common_words = {
        "this", "that", "blossom", "from", "have", "just",
        "like", "your", "what", "about", "with", "which",
        "will", "would", "could", "should", "really", "only",
        "still", "into", "some", "more", "them", "they",
        "then", "than", "when", "where", "been", "being",
        "those", "these", "also"
    }

    keywords = [w for w in set(words) if w not in common_words]
    selected = keywords[:max_keywords]
    print(f"🔍 Extracted keywords: {selected}")
    return selected
