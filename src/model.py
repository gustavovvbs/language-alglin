import random
import numpy as np
import autograd.numpy as np_
from utils import one_hot_encode, softmax, loss_batch
from data import load_data, build_vocab, encode_data, generate_skip_pairs
from autograd import grad, jacobian, hessian

def initialize_weights(vocab_size: int, embedding_dim: int) -> tuple:
    """
    inicializa os pesos do modelo de forma aleatoria
    
    :param vocab_size: tamanho do vocabulario
    :param embedding_dim: tamanho do embedding
    :return: pesos da camada de entrada e da camada de saida
    """
    w1 = np.random.uniform(-0.5, 0.5, (vocab_size, embedding_dim)).astype(np.float32)
    w2 = np.random.uniform(-0.5, 0.5, (embedding_dim, vocab_size)).astype(np.float32)
    return w1, w2

grad_w1 = grad(loss_batch, 0)
grad_w2 = grad(loss_batch, 1)

def train_model(pairs: list[tuple[int, int]], vocab_size: int, embedding_dim: int,
                epochs: int = 50, batch_size: int = 64, lr: float = 1e-3) -> tuple:
    """
    treina o modelo 

    :param pairs: lista de pares pro skipgram
    :param epochs: numero de epocas
    :param batch_size: tamanho do batch
    :param lr: taxa de aprendizado
    :return: pesos da camada de entrada, da camada de saida e o historico de loss
    """
    w1, w2 = initialize_weights(vocab_size, embedding_dim)
    num_pairs = len(pairs)
    history = []

    for ep in range(1, epochs + 1):
        total = 0.0
        for s in range(0, num_pairs, batch_size):
            batch = pairs[s:s+batch_size]
            xb = np.stack([one_hot_encode(c, vocab_size) for c, _ in batch])
            yb = np.stack([one_hot_encode(t, vocab_size) for _, t in batch])

            loss = loss_batch(w1, w2, xb, yb)
            gradient_w1 = grad_w1(w1, w2, xb, yb)
            gradient_w2 = grad_w2(w1, w2, xb, yb)

            w1 -= lr * gradient_w1
            w2 -= lr * gradient_w2

            total += loss * len(batch)

        avg = total/num_pairs
        history.append(avg)
        print(f"Epoch {ep}/{epochs} - Loss: {avg:.4f}")

    return w1, w2, history



