# Sistema de Recomendação por Conteúdo — KNN

Sistema de recomendação de filmes usando **filtragem baseada em conteúdo** com **KNN (k=1)** e visualização 3D interativa.

## Ideia Geral

Cada filme é representado como um **vetor em um espaço multidimensional**, onde cada dimensão corresponde a um gênero (ação, comédia, drama, etc.) com um valor entre 0 e 1. Um usuário também é representado como um vetor no mesmo espaço. O KNN encontra o filme **mais próximo** (menor distância euclidiana) e o recomenda.

## Estrutura do Projeto

```
Recomendation System/
├── data.py       # 20 filmes com vetores de gênero
├── knn.py        # KNN manual com distância euclidiana
├── visualize.py  # PCA manual + gráfico 3D interativo (Plotly)
├── main.py       # Orquestração principal
└── README.md     # Este arquivo
```

---

## data.py

Contém a lista de **gêneros** usados como dimensões e os **20 filmes** com seus respectivos vetores.

```python
GENRES = [
    "ação", "comédia", "drama", "terror",
    "ficção científica", "romance", "suspense", "animação",
]
```

Cada filme é uma tupla `(nome, [v0, v1, ..., v7])` onde cada valor corresponde ao gênero na mesma posição da lista `GENRES`.

**Exemplos:**
- `Matrix Reloaded` tem vetor `[0.9, 0.0, 0.2, 0.0, 0.9, 0.0, 0.3, 0.0]` → ação=0.9, comédia=0.0, drama=0.2, ficção científica=0.9
- `Cyberpunk: Edgerunners` tem vetor `[0.9, 0.1, 0.7, 0.0, 0.9, 0.2, 0.6, 1.0]` → ação=0.9, drama=0.7, FC=0.9, animação=1.0
- `Tudo em todo lugar ao mesmo tempo` tem vetor `[0.8, 0.8, 0.9, 0.1, 0.9, 0.6, 0.4, 0.0]` → ação=0.8, comédia=0.8, drama=0.9, FC=0.9

---

## knn.py

Implementação manual do **K-Nearest Neighbors** com k=1.

### Distância Euclidiana

```python
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
```

Calcula a raiz quadrada da soma das diferenças ao quadrado entre cada dimensão dos dois vetores. É a **distância geométrica** entre dois pontos no espaço de gêneros.

### KNN

```python
def knn(data_vectors, data_labels, query_vector, k=1):
```

1. Calcula a distância do vetor de consulta (usuário) para **todos** os filmes
2. Ordena do menor para o maior
3. Retorna os `k` filmes mais próximos

---

## visualize.py

### PCA (Principal Component Analysis) — Manual

O PCA é usado porque nossos dados têm **8 dimensões** (um para cada gênero), mas só conseguimos visualizar em 3D. O PCA reduz a dimensionalidade **preservando o máximo de informação possível**.

#### Por que PCA?

Sem ele, teríamos que escolher apenas 3 gêneros para os eixos do gráfico, perdendo toda a informação dos outros 5. O PCA encontra **novos eixos** (componentes principais) que capturam a maior parte da **variância** (espalhamento) dos dados.

#### Passo a passo do PCA implementado:

**1. Centralizar os dados**
```python
X_centered = X - mean
```
Subtrai a média de cada gênero, centralizando os pontos na origem.

**2. Matriz de covariância**
```python
cov = np.cov(X_centered, rowvar=False)
```
Mede como os gêneros variam juntos. Por exemplo: filmes com muita ação tendem a ter também ficção científica? A covariância captura essas relações.

**3. Autovalores e autovetores**
```python
eigenvalues, eigenvectors = np.linalg.eig(cov)
```
- **Autovetores**: direções dos novos eixos (componentes principais)
- **Autovalores**: o quanto da variância total cada eixo explica

**4. Ordenar por importância**
```python
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]
```
Organiza os componentes do que mais explica a variação para o que menos explica.

**5. Projetar para 3D**
```python
return X_centered @ components
```
Multiplica os dados centralizados pelos 3 autovetores mais importantes, obtendo coordenadas em 3D.

### Transformação do Usuário

O vetor do usuário precisa ser transformado usando **o mesmo PCA** dos filmes:

```python
def transform_user(user_vector, data_mean, components):
    return (np.array(user_vector) - data_mean) @ components
```

### Plotly 3D

O gráfico mostra 4 elementos:

| Cor | Símbolo | Significado |
|-----|---------|-------------|
| 🔵 Azul | Círculo | Filmes na base |
| 🟢 Verde | Diamante | Preferências do usuário |
| 🔴 Vermelho | Quadrado | Filme recomendado |
| Cinza | Linha tracejada | Conexão usuário → recomendado |

O gráfico é **interativo**: pode girar, dar zoom e passar o mouse sobre os pontos para ver os nomes.

---

## main.py

Orquestra o fluxo completo:

1. Carrega os filmes de `data.py`
2. Monta o vetor do perfil do usuário (fixo no código)
3. Executa o KNN para encontrar o filme mais próximo
4. Exibe os resultados no terminal com uma barra visual
5. Aplica PCA para reduzir para 3D
6. Abre o gráfico interativo no navegador

### Perfil do Usuário Fixo

```python
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
```

Esse perfil (fã de ação, ficção científica e animação) faz com que o KNN recomende **Cyberpunk: Edgerunners** (distância ~0.26).

---

## Como Executar

```bash
source venv/bin/activate
python main.py
```

Dependências: `numpy` (álgebra linear) e `plotly` (gráfico 3D interativo).
