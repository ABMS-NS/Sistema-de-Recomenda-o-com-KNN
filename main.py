from data import GENRES, MOVIES
from knn import knn
from visualize import pca, plot_3d, transform_user

USER_PROFILE = {
    "ação": 0.9,
    "comédia": 0.2,
    "drama": 0.6,
    "terror": 0.0,
    "ficção científica": 0.9,
    "romance": 0.2,
    "suspense": 0.5,
    "animação": 0.8,
}


def main():
    labels, vectors = zip(*MOVIES)
    vectors = [list(v) for v in vectors]

    user_vector = [USER_PROFILE[g] for g in GENRES]

    recomendacoes = knn(vectors, labels, user_vector, k=1)
    filme, distancia = recomendacoes[0]

    print("\n=== SISTEMA DE RECOMENDAÇÃO POR CONTEÚDO ===")
    print("Perfil do usuário:")
    for genero, valor in USER_PROFILE.items():
        barra = "█" * int(valor * 20)
        print(f"  {genero:>20}: {valor:.2f}  {barra}")
    print("\nKNN (k=1) — Distância Euclidiana")
    print(f"  ✅ Filme recomendado: {filme}")
    print(f"  📏 Distância:        {distancia:.4f}")

    movie_coords, mean, components = pca(vectors, n_components=3)
    user_coord = transform_user(user_vector, mean, components)

    print("\nAbrindo gráfico 3D interativo no navegador...\n")
    plot_3d(movie_coords, list(labels), user_coord, "VOCÊ", filme)


if __name__ == "__main__":
    main()
