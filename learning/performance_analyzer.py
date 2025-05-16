import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

def plot_tsne_3d_of_a(a_trace, selected_epochs: list[int], perplexity=20):
    """
    Visualize 3D t-SNE of all learned 'a' vectors using color and marker shape per domain.
    """
    all_a = []
    all_labels = []

    for domain_id, a_array in enumerate(a_trace):
        valid_indices = [i for i in selected_epochs if i < len(a_array)]
        if not valid_indices:
            continue
        a_flat = a_array[valid_indices].reshape(len(valid_indices), -1)
        all_a.append(a_flat)
        all_labels.append(np.full(len(valid_indices), domain_id))

    X = np.concatenate(all_a, axis=0)
    Y = np.concatenate(all_labels, axis=0)

    # 3D t-SNE
    X_tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate='auto', init='pca', random_state=0).fit_transform(X)

    # DataFrame for coloring/styling
    df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'z': X_tsne[:, 2],
        'domain': Y.astype(str)
    })

    # Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    unique_domains = df['domain'].unique()
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', '+', 'x', '1', '2', '3', '4']
    palette = sns.color_palette("tab10", n_colors=len(unique_domains))

    for i, domain in enumerate(unique_domains):
        subset = df[df['domain'] == domain]
        ax.scatter(subset['x'], subset['y'], subset['z'],
                   label=f"Domain {domain}",
                   s=40,
                   marker=markers[i % len(markers)],
                   color=palette[i % len(palette)])

    ax.set_title("3D t-SNE of a")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_zlabel("t-SNE dim 3")
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()

def plot_tsne_of_a(a_trace, selected_epochs: list[int], perplexity=20):
    """
    Visualize t-SNE of all learned 'a' vectors using both color and marker shape for each domain.
    """
    all_a = []
    all_labels = []

    for domain_id, a_array in enumerate(a_trace):
        valid_indices = [i for i in selected_epochs if i < len(a_array)]
        if not valid_indices:
            continue
        a_flat = a_array[valid_indices].reshape(len(valid_indices), -1)
        all_a.append(a_flat)
        all_labels.append(np.full(len(valid_indices), domain_id))

    X = np.concatenate(all_a, axis=0)
    Y = np.concatenate(all_labels, axis=0)

    X_tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=0).fit_transform(X)

    # Build DataFrame for Seaborn
    df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'domain': Y.astype(str)  # ensure legend shows full domain ID
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='domain', style='domain', palette='tab10', s=60)

    plt.title("t-SNE of a")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="Domain", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout()

