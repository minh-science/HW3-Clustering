# Write your k-means unit tests here
from sklearn.cluster import KMeans as sk_kmeans
from cluster import KMeans as ml_kmeans
from cluster.utils import make_clusters 

def test_kmeans():
    # initialize variables
    x_clusters, x_labels = make_clusters()

    # initialize sklearn kmeans
    sk_out = sk_kmeans(n_clusters= 3).fit(x_clusters)

    # initialize my kmeans 
    ml_out = ml_kmeans(k=3)
    ml_out.fit(x_clusters)
    ml_predict = ml_out.predict(x_clusters)

    # assert centroids are equivalent
    centroid_diff = sk_out.cluster_centers_ - ml_out.centroids 
    # print(centroid_diff)    
    for i in centroid_diff:
        assert i[0] < 0.1
        assert i[1] < 0.1

    
test_kmeans()
