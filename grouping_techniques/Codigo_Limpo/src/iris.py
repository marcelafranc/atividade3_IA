import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import MinMaxScaler

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

    petal_length = X[:, 2]
    petal_width = X[:, 3]

    markers = ['o', '^', 's']

    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.scatter(petal_width[y == i], petal_length[y == i], marker=markers[i], label=iris.target_names[i], edgecolor='k', alpha=0.7)

    plt.xlabel('Petal Width (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title("Visualização do Dataset Iris sem Agrupamento")
    plt.legend(title='Species')
    plt.show()

def metodo_silhouette(X_normalized):
    scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_normalized)
        score = silhouette_score(X_normalized, kmeans.labels_)
        scores.append(score)

    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), scores, marker='o')
    plt.title('Silhouette Score para Determinação de k')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.show()

    best_k = scores.index(max(scores)) + 2 
    print(f"O melhor número de clusters (k) baseado no Silhouette Score é: {best_k}")
    return best_k

def hierarquicoIris():
    X = iris.data
    y = iris.target

    petal_length = X[:, 2]
    petal_width = X[:, 3]

    markers = ['o', '^', 's']
    species = ['Setosa', 'Versicolor', 'Virginica']
    
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

    n_clusters = metodo_silhouette(X)
    print(f"Número de clusters sugerido: {n_clusters}")

    cluster_colors = plt.cm.get_cmap("Set1", n_clusters)

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
        
        for j in range(n_clusters):
            cluster_indices = y_hr == j
            for k in range(3):
                species_indices = y == k
                axes[i//2, i%2].scatter(
                    petal_width[species_indices & cluster_indices], 
                    petal_length[species_indices & cluster_indices], 
                    marker=markers[k], 
                    color=cluster_colors(j),
                    edgecolor='k', alpha=0.7
                )
        
        axes[i//2, i%2].set_title(titles[i])
        axes[i//2, i%2].set_xlabel('Petal Width (cm)')
        axes[i//2, i%2].set_ylabel('Petal Length (cm)')
        axes[i//2, i%2].legend(species, title="Espécies", loc='upper right')
    
    plt.tight_layout()
    plt.show()

def tecnica_elbow(X, max_k=10):
    inertias = []
    k_values = range(2, max_k + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertias, marker='o', linestyle='-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Método do Cotovelo para Determinar k (Iris Dataset)')
    plt.show()

    dif_inertias = np.diff(inertias)
    k_best = np.argmax(dif_inertias) + 2
    return k_best

def particionalIris(num_iterations=1000):
    X = iris.data[:, [2, 3]]
    y = iris.target

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    k_optimal = tecnica_elbow(X_normalized)

    best_silhouette = -1
    
    best_labels = None
    best_centroids = None

    for i in range(num_iterations):
        kmeans = KMeans(n_clusters=k_optimal, random_state=None, n_init=1, init='random')
        kmeans.fit(X_normalized)
        
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        print(f"Iteração {i + 1}, Silhouette Score: {silhouette_avg}")

        if silhouette_avg > best_silhouette:
            best_silhouette = silhouette_avg
            best_labels = kmeans.labels_
            best_centroids = kmeans.cluster_centers_
            best_trial = i
       
    print(f"Melhor Silhueta: {best_silhouette}, Obtido no trial {best_trial + 1}")

    best_centroids_original = scaler.inverse_transform(best_centroids)

    cluster_colors = plt.cm.viridis(np.linspace(0, 1, k_optimal))

    markers = ['^', 'o', 's']
    species = ['Setosa', 'Versicolor', 'Virginica']

    plt.figure(figsize=(8, 6))

    for i, species_label in enumerate(np.unique(y)):
        species_indices = np.where(y == species_label)[0]
        X_species = scaler.inverse_transform(X_normalized[species_indices])
        plt.scatter(X_species[:, 0], X_species[:, 1], 
                    c=cluster_colors[best_labels[species_indices]],
                    marker=markers[i], label=species[i], alpha=0.7)

    plt.scatter(best_centroids_original[:, 0], best_centroids_original[:, 1], s=200, c=cluster_colors, marker='X', label='Centroids')

    plt.xlabel('PetalWidth (cm)')
    plt.ylabel('PetalLength (cm)')
    plt.title(f'Agrupamento particional do dataset IRIS, 1000 iterações')

    plt.legend(loc='best')
    plt.show()