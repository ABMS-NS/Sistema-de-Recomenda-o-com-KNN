import numpy as np
import plotly.graph_objects as go


def pca(X, n_components=3):
    X = np.array(X, dtype=float)
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    components = eigenvectors[:, :n_components].real
    return X_centered @ components, mean, components


def transform_user(user_vector, data_mean, components):
    return (np.array(user_vector, dtype=float) - data_mean) @ components


def plot_3d(movie_coords, movie_labels, user_coord, user_label, recommended_label):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=movie_coords[:, 0],
        y=movie_coords[:, 1],
        z=movie_coords[:, 2],
        mode="markers+text",
        text=movie_labels,
        textposition="top center",
        marker=dict(size=8, color="blue", opacity=0.7),
        name="Filmes",
        hovertext=movie_labels,
    ))

    fig.add_trace(go.Scatter3d(
        x=[user_coord[0]],
        y=[user_coord[1]],
        z=[user_coord[2]],
        mode="markers+text",
        text=["<b>VOCÊ</b>"],
        textposition="top center",
        marker=dict(size=14, color="green", symbol="diamond",
                     line=dict(width=2, color="darkgreen")),
        name="Preferências (você)",
    ))

    recommended_idx = movie_labels.index(recommended_label)
    rec_coord = movie_coords[recommended_idx]

    fig.add_trace(go.Scatter3d(
        x=[rec_coord[0]],
        y=[rec_coord[1]],
        z=[rec_coord[2]],
        mode="markers+text",
        text=[f"<b>★ {recommended_label}</b>"],
        textposition="top center",
        marker=dict(size=14, color="red", symbol="square",
                     line=dict(width=2, color="darkred")),
        name="★ Recomendado",
    ))

    fig.add_trace(go.Scatter3d(
        x=[user_coord[0], rec_coord[0]],
        y=[user_coord[1], rec_coord[1]],
        z=[user_coord[2], rec_coord[2]],
        mode="lines",
        line=dict(color="gray", width=2, dash="dash"),
        name="Ligação",
    ))

    fig.update_layout(
        title="Sistema de Recomendação por Conteúdo — KNN (k=1)",
        scene=dict(
            xaxis_title="Componente Principal 1",
            yaxis_title="Componente Principal 2",
            zaxis_title="Componente Principal 3",
        ),
        width=1000,
        height=800,
        hovermode="closest",
    )

    fig.show()
