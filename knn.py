import math


def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn(data_vectors, data_labels, query_vector, k=1):
    distances = []
    for vec, label in zip(data_vectors, data_labels):
        d = euclidean_distance(query_vector, vec)
        distances.append((label, d))
    distances.sort(key=lambda x: x[1])
    return distances[:k]
