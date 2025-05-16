from model import train_model
import random
from data import load_data, build_vocab, encode_data, generate_skip_pairs
from evalm import nearest_embed, plot_sentiment_cluster
import numpy as np
import matplotlib.pyplot as plt

reviews = load_data(num_reviews =4000)
word2idx, idx2word = build_vocab(reviews)
tokens_idx = encode_data(reviews, word2idx)
pairs = generate_skip_pairs(tokens_idx, window_size=5)


# treinamento (descomente a linha abaixo para treinar o modelo do zero)
# train_model(pairs, vocab_size=len(word2idx), embedding_dim=100)

w1 = np.load("src/weights/w1.npy")
w2 = np.load("src/weights/w2.npy")

for w in ["bad", "terrible"]:
    if w in word2idx:
        print(f"Palavra: {w}")
        print("Vizinhos mais proximos:")
        for neighbor, sim in nearest_embed(w1, w, word2idx, idx2word):
            print(f"  {neighbor}: {sim:.4f}")
    else:
        print(f"Palavra {w} não encontrada no vocabulario")

plot_sentiment_cluster(w1, word2idx)
plt.show()
