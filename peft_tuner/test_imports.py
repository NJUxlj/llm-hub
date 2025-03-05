try:
    from transformers import AutoModel, AutoTokenizer
    from sentencepiece import SentencePieceProcessor
    print("Imports successful")
except Exception as e:
    print(f"Error: {e}")