import numpy as np
from utils import pca_2d
import matplotlib.pyplot as plt

def nearest_embed(w1: np.ndarray, word: str, word2idx: dict[str, int], idx2word: dict[int, str], top_n: int = 5):
    """
    encontra os top_n vizinhos mais proximos de uma palavra no embedding

    :param w1: pesos da camada de entrada
    :param word: palavra
    :param word2idx: dicionario de indices
    :param idx2word: dicionario de palavras
    :param top_n: numero de vizinhos mais proximos
    :return: lista de vizinhos mais proximos
    """
    if word not in word2idx:
        print(f"Palavra {word} não encontrada no vocabulario")
        return []
    
    idx = word2idx[word]
    # o embedding eh a linha idx da matriz w1
    v = w1[idx]

    #uso a similaridade cosseno como metrica de similaridade
    sims = (w1 @ v) / (np.linalg.norm(w1, axis = 1) * np.linalg.norm(v) + 1e-8) 
    nearest = np.argsort(-sims)[:top_n+1] # o +1 eh pq a palavra eh a primeira
    return [(idx2word[i], sims[i]) for i in nearest if i != idx][:top_n]


def plot_sentiment_cluster(w1: np.ndarray, word2idx: dict[str, int]) -> None:
    """
    plota os embeddings de palavras em um grafico 2d usando PCA pra reduzir a dimensionalidade
    
    :param w1: pesos da camada de entrada
    :param word2idx: dicionario de indices
    :return: None
    """

    pos = ["great", "good"]
    neg = ["terrible", "bad"]

    words = [w for w in pos + neg if w in word2idx and word2idx[w] < w1.shape[0]]
    if not words:
        print("Nenhuma palavra encontrada no embedding.")
        return
    vecs = np.stack([w1[word2idx[w]] for w in words])
    coordinates = pca_2d(vecs)
    plt.figure(figsize=(6, 6))
    plt.scatter(coordinates[:len(pos), 0], coordinates[:len(pos), 1], c="green", label="Positivo")
    plt.scatter(coordinates[len(pos):, 0], coordinates[len(pos):, 1], c="red", label="Negativo")
    for w, (x, y) in zip(words, coordinates):
        plt.annotate(w, (x, y))
    plt.title("Clusters de Sentimento (PCA)")
    plt.legend()
    