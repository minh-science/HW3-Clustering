# write your silhouette score unit tests here
import numpy as np 
from sklearn.cluster import KMeans as sk_kmeans
from cluster import KMeans as ml_kmeans
from cluster.utils import make_clusters 
from cluster import Silhouette
from sklearn.metrics import silhouette_score, silhouette_samples



def test_silhouette():
    # initialize variables
    clusters, labels = make_clusters()

    # # initialize sklearn kmeans
    # sk_fit_out = sk_kmeans(n_clusters= 3).fit(x_clusters)
    # sk_out = sk_kmeans(n_clusters= 3).fit_predict(x_clusters)

    # initialize my kmeans 
    ml_out = ml_kmeans(k=3)
    ml_out.fit(clusters)

    ml_predict = ml_out.predict(clusters)

    # my silhoutette scores
    ml_scores = Silhouette().score(X= clusters, y= ml_predict)
    # print(ml_scores)
    ml_score = np.mean(ml_scores)
    print(ml_score)

    # sklearn silhouette scores
    sk_score = silhouette_score(clusters, ml_predict)
    print(sk_score)
    print(np.mean(silhouette_samples(X = clusters, labels = ml_predict)))


test_silhouette()


