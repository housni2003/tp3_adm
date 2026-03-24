import json
import os

notebook_path = 'c:/Users/hi/code/polytech/adm/tp3/tp3.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_agg = """# 6. Agglomerative Clustering avec trois méthodes (ward, average, single)
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib

linkages = ['ward', 'average', 'single']
colors = ['red', 'yellow', 'blue']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, linkage in enumerate(linkages):
    agg_cluster = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    labels_agg = agg_cluster.fit_predict(X_scaled)
    
    ax = axes[idx]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_agg, cmap=matplotlib.colors.ListedColormap(colors))
    
    for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
        ax.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points', fontsize=6)
        
    ax.set_title(f"{linkage.capitalize()} Linkage")
    ax.set_xlabel('Composante Principale 1')
    ax.set_ylabel('Composante Principale 2')
    ax.grid(True)

plt.tight_layout()
plt.savefig('agglomerative_clusters_villes.png')
print("Graphique sauvegardé dans agglomerative_clusters_villes.png")
plt.show()
"""

code_sil = """# 7. Détermination de la meilleure partition avec le critère Silhouette
from sklearn.metrics import silhouette_score
import numpy as np

range_n_clusters = range(2, 11)

silhouette_kmeans = []
silhouette_agg = []

for n_clusters in range_n_clusters:
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels_k = kmeans.fit_predict(X_scaled)
    silhouette_kmeans.append(silhouette_score(X_scaled, cluster_labels_k))
    
    # Agglomerative Clustering (ward is standard)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    cluster_labels_a = agg.fit_predict(X_scaled)
    silhouette_agg.append(silhouette_score(X_scaled, cluster_labels_a))

plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_kmeans, marker='o', label='KMeans')
plt.plot(range_n_clusters, silhouette_agg, marker='s', label='Agglomerative (Ward)')
plt.xlabel('Nombre de clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Nombre de clusters')
plt.legend()
plt.grid(True)
plt.savefig('silhouette_scores.png')
print("Graphique sauvegardé dans silhouette_scores.png")
plt.show()

best_k = range_n_clusters[np.argmax(silhouette_kmeans)]
best_a = range_n_clusters[np.argmax(silhouette_agg)]
print(f"Meilleure partition pour KMeans: {best_k} clusters avec un score de {max(silhouette_kmeans):.4f}")
print(f"Meilleure partition pour AgglomerativeClustering: {best_a} clusters avec un score de {max(silhouette_agg):.4f}")
"""

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))]
    }

has_agg = any('Agglomerative Clustering avec trois méthodes' in "".join(cell.get('source', [])) for cell in nb['cells'])
has_sil = any('Détermination de la meilleure partition' in "".join(cell.get('source', [])) for cell in nb['cells'])

if not has_agg:
    nb['cells'].append(create_code_cell(code_agg))
if not has_sil:
    nb['cells'].append(create_code_cell(code_sil))

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook successfully modified using JSON.")
