import numpy as np

def distance(x, X):
    return np.sqrt(np.sum((x - X) ** 2))

def KNN(X, Y, x, K=5):
    x = x.reshape((10000,))
    m = X.shape[0]
    vals = []

    for i in range(m):
        xi = X[i].reshape((10000,))
        dist = distance(x, xi)
        vals.append((dist, int(Y[i])))

    vals = sorted(vals, key=lambda z: z[0])[:K]
    vals = np.asarray(vals)

    # Vote for the most common class
    classes, counts = np.unique(vals[:, 1], return_counts=True)
    return classes[counts.argmax()]
