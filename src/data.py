import re
import random
import numpy as np
from datasets import load_dataset
from collections import Counter

def load_data(num_reviews=500) -> list[str]:
    """
    carrega o dataset de reviews do IMDB e retorna os reviews e as labels

    :param num_reviews: número de reviews a serem carregados
    :return: uma lista com as reviews separadas por palavra
    """

    ds = load_dataset("imdb", split = "train")
    tokens_raw = []
    for text in ds["text"][:500]:
        tokens_raw.extend(re.findall(r"[A-Za-z]+[\w^']*|[\w^']*[A-Za-z]+[\w^']*", text.lower()))

    print(f"{len(tokens_raw)} tokens tokenizados")

    return tokens_raw

def build_vocab(tokens_raw: list, crop_size = 5000):
    """
    Cria um vocabulario a partir dos tokens, e retorna o vocabulario e o dicionario de indices

    :param tokens_raw: lista de tokens
    :param crop_size: tamanho do vocabulario
    :return: vocabulario e dicionario de indices
    """

    vocab_count = Counter(tokens_raw)
    most_common = vocab_count.most_common(crop_size)
    word2idx = { w:i for i, (w, _) in enumerate(most_common) }
    word2idx["<unk>"] = len(word2idx)

    idx2word = { i:w for w, i in word2idx.items() }

    vocab_size = len(word2idx)

    print(f"Vocabulario criado com {vocab_size} palavras")

    return word2idx, idx2word

def encode_data(tokens_raw: list[str], word2idx: dict[str, int]) -> list[list[int]]:
    """
    encoda o dataset de texto para uma sequencia de indices
    
    :param tokens_raw: lista de tokens
    :param word2idx: dicionario de indices
    :return: lista de indices
    """

    tokens_idx = [word2idx.get(w, word2idx["<unk>"]) for w in tokens_raw]
    return tokens_idx


def generate_skip_pairs(tokens_idx: list[int], window_size: int = 5) -> list[tuple[int, int]]:
    """
    gera os pares skip-gram a partir dos indices

    :param tokens_idx: lista de indices
    :param window_size: tamanho da janela
    :return: lista de pares skip-gram
    """

    skip_pairs = []
    for i, center in enumerate(tokens_idx):
        start = max(0, i - window_size)
        end = min(len(tokens_idx), i + window_size + 1)
        for j in range(start, end):
            if j != i:
                skip_pairs.append((center, tokens_idx[j]))

    random.shuffle(skip_pairs)
    print(f"{len(skip_pairs)} pares gerados")

    return skip_pairs