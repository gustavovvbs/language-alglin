import numpy as np
import autograd.numpy as np_

def one_hot_encode(idx: int, vocab_size: int) -> np.array:
    """
    faz o one hot encoding de um indice
    
    :param idx: indice
    :param vocab_size: tamanho do vocabulario
    """
    one_hot = np.zeros(vocab_size, dtype=np.float32)
    one_hot[idx] = 1.0
    return one_hot


def softmax(x: np_.ndarray) -> np_.ndarray:
    """
    calcula a softmax de um vetor
    
    :param x: vetor
    :return: vetor softmax
    """
    x = x - np_.max(x, axis=-1, keepdims=True)
    e = np_.exp(x)
    return e / (np_.sum(e, axis=-1, keepdims=True) + 1e-10)


def loss_batch(w1: np_.ndarray, w2: np_.ndarray, xb: np_.ndarray,
               yb: np_.ndarray) -> np_.ndarray:
    """
    calcula a loss (entropia cruzada) de um batch

    :param w1: pesos da camada de entrada
    :param w2: pesos da camada de saida
    :param xb: batch de entrada
    :param yb: batch de saida
    :return: loss
    """
    a1 = xb @ w1  # (batch_size, vocab_size)
    a2 = a1 @ w2  # (batch_size, vocab_size)
    yhat = softmax(a2)  # (batch_size, vocab_size)
    loss = -np_.sum(yb * np_.log(yhat + 1e-10)) / xb.shape[0]  # (batch_size,)
    return loss


def pca_2d(X: np.ndarray) -> np.ndarray:
    """
    reduz a dimensionalidade da matriz para 2D usando PCA, e retorna as coordenadas 2D(x, y)

    :param X: matriz de dados
    :return: matriz de dados reduzida
    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt.T[:, :2]  # (n_samples, 2)