import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from models_helpers import *



## loading the data
X, y=load_data(0)
plot_images(X, 5,5, 'Sample')
plt.gcf().set_size_inches(7, 8)
plt.show()


## fitting the pca model on the data
pca=PCA()
pca.fit(X)
plot_images(pca.evectors.T, 5, 5, 'PC')
plt.gcf().set_size_inches(7, 8)
plt.show()


# variance vs pc plot
plt.plot((pca.evalues/pca.evalues.sum()).cumsum()*100)
ax=plt.gca()
ax.set_yticks(np.arange(105, step=5))
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('% of Variance')
ax.set_title('Variance vs Principal Components');
plt.show()


## reconstruct an image using different number of pcs
plot_for_pcs(3, 4, X[702], [1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, None], pca)
plt.gcf().set_size_inches(7, 6)
plt.show()


# reconstruct some samples usinf 300 pcs
X_compressed=pca.transform(X, 300)
X_recon=pca.reconstruct(X_compressed, 300)
plot_images(X_recon, 5, 5, 'Sample')
plt.gcf().set_size_inches(7, 8)
plt.show()




### question2, loading the 
X=pd.read_csv('cm_dataset_2.csv', header=None).values


## plotting for different initializations
for state in [0, 100, 200, 400, 500]:
    km=KMeans(2, state, 1000)
    km.fit(X)
    km.plot_clusters(X, ['red', 'blue'])
    plt.gcf().set_size_inches(12, 6)
    plt.tight_layout()
    plt.show()

for cluster in [2, 3, 4, 5]:
    km=KMeans(cluster, 3, 1000)
    km.fit(X)
    km.plot_voronoi(X, ['red', 'gray', 'blue', 'yellow', 'green'])
    plt.show()