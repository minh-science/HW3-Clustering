# write your silhouette score unit tests here
from sklearn.cluster import KMeans as sk_kmeans
from cluster import KMeans as ml_kmeans
from cluster.utils import make_clusters 
from cluster import Silhouette
from sklearn.metrics import silhouette_score



def test_silhouette():
    x_clusters, x_labels = make_clusters()

    sk_out = sk_kmeans(n_clusters= 3).fit(x_clusters)

    ml_out = ml_kmeans(k=3)
    ml_out.fit(x_clusters)
    ml_predict = ml_out.predict(x_clusters)

    scores = Silhouette().score(x_clusters, ml_predict)