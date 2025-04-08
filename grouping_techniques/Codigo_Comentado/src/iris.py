import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler

# Carregar o dataset Iris
iris = load_iris()

def pairplotIris():

    
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target

    sns.pairplot(df_iris, hue='target', diag_kind='hist', palette='tab10')
    plt.suptitle("Pairplot Completo do Dataset Iris", y=1.02)
    plt.show()

def printIris():
    X = iris.data
    y = iris.target

    # Normalizar os dados ---> nao vou usar aqui
    #scaler = MinMaxScaler()
    #X_normalized = scaler.fit_transform(X)

    # Atributos para o gráfico
    petal_length = X[:, 2]  # Petal Length
    petal_width = X[:, 3]   # Petal Width

    # Formas geométricas baseadas na espécie
    markers = ['o', '^', 's']  # 'o' para Setosa, '^' para Versicolor, 's' para Virginica

    # Plotando o gráfico sem agrupamento
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(petal_width[y == i], petal_length[y == i], marker=markers[i], label=iris.target_names[i], edgecolor='k', alpha=0.7)

    plt.xlabel('Petal Width (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title("Visualização do Dataset Iris sem Agrupamento")
    plt.legend(title='Species')
    plt.show()


# Método para determinar o número de clusters usando o Silhouette Score!!!!!!
def metodo_silhouette(X_normalized):
    scores = []
    # testa de 2 a 10
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_normalized)
        score = silhouette_score(X_normalized, kmeans.labels_)
        scores.append(score)

    # Plotar o grafico do silhouete !!
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), scores, marker='o')
    plt.title('Silhouette Score para Determinação de k')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()

    # encontrar o k com o melhor silhouete (definicao)
    best_k = scores.index(max(scores)) + 2  # comeca em 2
    print(f"O melhor número de clusters (k) baseado no Silhouette Score é: {best_k}")
    return best_k

# Agrupamento Hierárquico da Iris
def hierarquicoIris():
    X = iris.data
    y = iris.target

    # Normalizar os dados --> nao vou usar aqui
    #scaler = MinMaxScaler()
    #X_normalized = scaler.fit_transform(X)

    # Atributos para o gráfico
    petal_length = X[:, 2]  # Petal Length
    petal_width = X[:, 3]   # Petal Width

    # Formas geométricas baseadas na espécie
    markers = ['o', '^', 's']  # 'o' para Setosa, '^' para Versicolor, 's' para Virginica
    species = ['Setosa', 'Versicolor', 'Virginica']
    
    # 1- DENDROGRAMAS PRA DECIDIR ONDE TRACAR A LINHA (DEFINIR K)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    linkages = ["ward", "complete", "average", "single"]
    titles = [
        "Método Ward",
        "Método Complete Linkage",
        "Método Average Linkage",
        "Método Single Linkage"
    ]
    
    for i, linkage_method in enumerate(linkages):
        Z = linkage(X, method=linkage_method)
        dendrogram(Z, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(titles[i])
        axes[i//2, i%2].set_xlabel("Amostras")
        axes[i//2, i%2].set_ylabel("Distância")
    
    plt.tight_layout()
    plt.show()

    # 2 - DEFINIR O NUMERO DE CLUSTERS (DEFINE K)
    n_clusters = metodo_silhouette(X)
    print(f"Número de clusters sugerido: {n_clusters}")

    # Gerando uma paleta de cores para os clusters
    cluster_colors = plt.cm.get_cmap("Set1", n_clusters)  # Usando uma paleta com 3 cores

    # 3 - GRAFICOS DE DISPERSAO USANDO K DEFINIDO (MELHOR = 3)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    titles = [
        "Dendrograma - Método Ward",
        "Dendrograma - Método Complete Linkage",
        "Dendrograma - Método Average Linkage",
        "Dendrograma - Método Single Linkage"
    ]
    
    for i, linkage_method in enumerate(linkages):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        y_hr = clustering.fit_predict(X)
        
        # Plotando os clusters com cores diferentes (representando os clusters)
        for j in range(n_clusters):
            cluster_indices = y_hr == j  # Índices dos pontos no cluster j
            # Para cada cluster, plotar os pontos com o formato correspondente à espécie
            for k in range(3):  # Usando os diferentes formatos para cada espécie
                species_indices = y == k  # Índices dos pontos da espécie k
                axes[i//2, i%2].scatter(
                    petal_width[species_indices & cluster_indices], 
                    petal_length[species_indices & cluster_indices], 
                    marker=markers[k], 
                    color=cluster_colors(j),  # Cor do cluster
                    edgecolor='k', alpha=0.7
                )
        
        axes[i//2, i%2].set_title(titles[i])
        axes[i//2, i%2].set_xlabel('Petal Width (cm)')
        axes[i//2, i%2].set_ylabel('Petal Length (cm)')
        axes[i//2, i%2].legend(species, title="Espécies", loc='upper right')  # Apenas as espécies com seus formatos
    
    plt.tight_layout()
    plt.show()



# Função para a Técnica do Cotovelo
def tecnica_elbow(X, max_k=10):
    inertias = []
    k_values = range(2, max_k + 1)  # Começamos em 2 porque k=1 não faz sentido para clustering

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)  # Soma dos erros quadráticos dentro dos clusters

    # Plotando a curva do método do cotovelo
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertias, marker='o', linestyle='-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Método do Cotovelo para Determinar k (Iris Dataset)')
    plt.show()

    # Determina o melhor k pelo maior "joelho" da curva
    dif_inertias = np.diff(inertias)
    k_best = np.argmax(dif_inertias) + 2  # +2 pois o range começa em 2
    return k_best

# Função de K-means iterativo






def particionalIris(num_iterations=1000):
    X = iris.data[:, [2, 3]]  # PetalLength e PetalWidth (atributos que queremos usar)
    y = iris.target  # As espécies da planta

    # Normalizar os dados
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Encontrar o número ótimo de clusters com a técnica do cotovelo
    k_optimal = tecnica_elbow(X_normalized)

    # Inicializar variáveis para armazenar o melhor modelo
    best_silhouette = -1
    
    best_labels = None
    best_centroids = None

    # Iterar várias vezes para minimizar o erro quadrático
    for i in range(num_iterations):
        kmeans = KMeans(n_clusters=k_optimal, random_state=None, n_init=1, init='random')  # n_init=1 para inicialização aleatória
        kmeans.fit(X_normalized)

        # Calcular o Silhouette Score para esta execução
        
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        print(f"Iteração {i + 1}, Silhouette Score: {silhouette_avg}")

        
        

        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_labels = kmeans.labels_
            best_centroids = kmeans.cluster_centers_
            best_trial = i
       
    
    print(f"Melhor Silhueta: {best_silhouette}, Obtido no trial {best_trial + 1}")

    # Desnormalizar os centróides
    best_centroids_original = scaler.inverse_transform(best_centroids)

    # Definir uma paleta de cores fixa para os clusters
    cluster_colors = plt.cm.viridis(np.linspace(0, 1, k_optimal))

    # Definir os símbolos para as espécies (Triângulo, Círculo, Quadrado)
    markers = ['^', 'o', 's']  # Triângulo, Círculo, Quadrado
    species = ['Setosa', 'Versicolor', 'Virginica']  # As espécies no dataset Iris

    # Plotar o resultado final
    plt.figure(figsize=(8, 6))

    # Plotar os pontos com a cor de cada cluster e formato baseado na espécie
    for i, species_label in enumerate(np.unique(y)):
        species_indices = np.where(y == species_label)[0]
        # Desnormalizar os pontos para exibição
        X_species = scaler.inverse_transform(X_normalized[species_indices])
        plt.scatter(X_species[:, 0], X_species[:, 1], 
                    c=cluster_colors[best_labels[species_indices]],  # Usando as cores fixas para clusters
                    marker=markers[i], label=species[i], alpha=0.7)

    # Plotando os centróides (com as cores fixas de seus respectivos clusters)
    plt.scatter(best_centroids_original[:, 0], best_centroids_original[:, 1], s=200, c=cluster_colors, marker='X', label='Centroids')

    # Configurações do gráfico
    plt.xlabel('PetalWidth (cm)')
    plt.ylabel('PetalLength (cm)')
    plt.title(f'Agrupamento particional do dataset IRIS, 1000 iterações')

    # Exibir apenas a legenda das espécies (com os formatos) e os centróides
    plt.legend(loc='best')
    plt.show()