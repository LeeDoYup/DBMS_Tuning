#-*- coding: utf-8 -*-
'''
Created on Jul 4, 2016

@author: dvanaken
'''
import numpy as np
import tensorflow as tf
import os.path
from scipy.spatial.distance import cdist
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

from cluster import KMeans_, KSelection, GaussianMixture_, Spectral_clustering_, Hierarchy_clustering_
from matrix import Matrix
from preprocessing import get_shuffle_indices
from matplotlib import pyplot as plt
import pdb
from util import stdev_zero, read_wine, read_wine_quality



OPT_METRICS = ["fixed acidity", "volatile acidity"]
col = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

REQUIRED_VARIANCE_EXPLAINED = 90




def main():
    
    
    #------------------------------------------
    # Load raw value of X,y and make it Matrix
    #------------------------------------------
    x_columnlabels = np.array(['fixed acidity','volatile acidity','critic acid','residual sugar','chlorides', 
                                'free sulfur dioxide','total sulfur dioxide','density', 'pH', 'sulphates',
                                'alcohol'])
    y, x = read_wine_quality()
    y = y.reshape([-1, 1]) # expand dimension
    
    y_columnlabels = np.arange(y.shape[1])
    rowlabels = np.arange(x.shape[0])
    X = Matrix(x,rowlabels,x_columnlabels)
    y = Matrix(y,rowlabels,y_columnlabels)
    

    #------------------------------------
    # Select dimension reduction method.
    # 1. Factor Analysis
    # 2. Principal Component Analysis
    # 3. Auto Encoder
    #------------------------------------
    
    components, components_columnlabels = run_factor_analysis(X)
    #components, components_columnlabels = run_PCA(X)
    #components, components_columnlabels = run_AE(X)


    #--------------------------------------------
    # Select clustering method. 
    # 1. Kmeans
    # 2. Mixture of Gaussian
    # 3. Spectral clustering
    # 4. Hierarchical clustering
    #--------------------------------------------

    run_kmeans(components, components_columnlabels)
    #run_mog(components, components_columnlabels)
    #run_spectral(components, components_columnlabels)
    #run_hierarchy(components, components_columnlabels)

    return

def run_AE(X,epoch=100,batch_size=1,num_hidden=3):
    """Execute Auto Encoder.
    Arg
      X : X data with column, row label. (Matrix, data : [num_features(1599),num_samples(11)])
    Return
      components : Result of auto encoder. 
                   (Numpy array, [# of features, # of components]) 
      components_columnlabels : Labels for each componenets. (Numpy array, [# of features,]) 
    """
    import math

    
    # Get number of features and number of data. Each are 1599 and 11. Number of data is too small.
    num_features = X.data.shape[0]
    num_data = X.data.shape[1]

    # Preprocess : standardization
    standardizer = StandardScaler()
    X_data = standardizer.fit_transform(X.data.T)

    def model(X_placeholder):
        """Tensorflow Graph of learning model.
        Arg
          X_placeholder : Placeholder for X data. (Tensor, [batch_size,# of features])
          num_hidden : number of hidden units. 
        Return
          reconstruction : Reconstructed X data. same shape with X_placeholder. 
                           (Tensor, [batch_size,# of features])
          hidden : Hidden representaiton of data which is dimension reduced data.
          loss : Mean square error of reconstruction (Tensor, scalar)
        """

        # Declare parameters.
        W_encode = tf.get_variable("W_e",initializer=tf.truncated_normal([num_features,num_hidden],stddev=0.01))
        b_encode = tf.get_variable("b_e",initializer=tf.truncated_normal([num_hidden],stddev=0.01))
        W_decode = tf.get_variable("W_d",initializer=tf.truncated_normal([num_hidden,num_features],stddev=0.01))
        b_decode = tf.get_variable("b_d",initializer=tf.truncated_normal([num_features],stddev=0.01))

        # Build model.
        hidden = tf.tanh(tf.matmul(X_placeholder,W_encode)+b_encode)
        reconstruction = tf.matmul(hidden,W_decode)+b_decode
        
        # Calculate loss
        loss = tf.reduce_mean(tf.squared_difference(X_placeholder,reconstruction))

        return reconstruction, hidden, loss
    
    
    sess = tf.Session()

    # Graph Construction.
    X_placeholder = tf.placeholder(tf.float32, shape = [None,num_features]) 
    #reconstruction,hidden,loss = model(X_placeholder)
    
    # Declare parameters.
    W_encode = tf.get_variable("W_e",initializer=tf.truncated_normal([num_features,num_hidden],stddev=0.1))
    b_encode = tf.get_variable("b_e",initializer=tf.truncated_normal([num_hidden],stddev=0.1))
    W_decode = tf.get_variable("W_d",initializer=tf.truncated_normal([num_hidden,num_features],stddev=0.1))
    b_decode = tf.get_variable("b_d",initializer=tf.truncated_normal([num_features],stddev=0.1))

    # Build model.
    hidden = tf.tanh(tf.matmul(X_placeholder,W_encode)+b_encode)
    reconstruction = tf.matmul(hidden,W_decode)+b_decode
    
    # Calculate loss
    loss = tf.reduce_mean(tf.squared_difference(X_placeholder,reconstruction))

    # Generate Optimizer. 
    optimizer = tf.train.AdamOptimizer(learning_rate = 1e-6)
    train = optimizer.minimize(loss)
    
    # Varible Initialization.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Copy X data. In order to shuffle its order. The original data will be used to calculate final result.
    X_copy = np.copy(X_data)

    # Run learning iterations.
    num_batch = int(math.ceil(num_data/float(batch_size)))
    for i in range(epoch):
        
        # Shuffle the order of data. For Stochastic Gradient Descent.
        random_order = np.random.permutation(num_data)
        X_data = X_data[random_order]
        
        loss_list = []
        for j in range(num_batch):
            next_batch = X_data[j*batch_size:(j+1)*batch_size]
            _, recon_value, components, loss_value = sess.run([train,reconstruction,hidden,loss],
                                                         feed_dict = {X_placeholder:next_batch})
            loss_list.append(loss_value)
    
        mean_loss = np.mean(loss_list)
        print "{}th loss : {}".format(i+1,mean_loss)
            
    # Calculate final component value
    final_components,recon_final = sess.run((hidden,reconstruction),feed_dict = {X_placeholder:X_copy})
    recon_final = standardizer.inverse_transform(recon_final)


    print "final components : {}, reconstruction : {}, original components : {}".format(final_components[0],
                                                                                        recon_final[0],
                                                                                        X.data.T[0])

    return final_components, X.columnlabels
      


def run_PCA(X):
    """Execute Principal Component Analysis. 
    Arg
      X : X data with column, row label. (Matrix)
    
    Return
      components : Result of pca in variance descending order. 
                   (Numpy array, [# of features, # of components]) 
      components_columnlabels : Labels for each componenets. (Numpy array, [# of features,]) 
    """
    
    
    #--------------
    # Execute PCA.
    #--------------
    
    pca = PCA()
    pca.fit(X.data)
    
    #------------------------------
    # Determine number of factors.
    #------------------------------
    
    # Only nonzero components should be considered.
    pca_mask = np.sum(pca.components_ != 0.0, axis=1) > 0.0

    # Select number of components which can explain REQUIRED_VARIANCE_EXPLAINED percent of variance.
    variances = pca.explained_variance_ratio_
    variances_explained_percent = np.array([np.sum(variances[:i+1]) * 100 for i in range(variances.shape[0])])
    component_cutoff = np.count_nonzero(variances_explained_percent < REQUIRED_VARIANCE_EXPLAINED) + 1
    component_cutoff = min(component_cutoff, 10)
    
    #print variances.
    print "component cutoff: {}".format(component_cutoff)
    for i,var in enumerate(variances):
        print i, var, np.sum(variances[:i+1]), np.sum(variances[:i+1])


    #----------------
    # Postprecessing
    #----------------
    
    # Standardization
    components = np.transpose(pca.components_[:component_cutoff]).copy()
    print "components shape: {}".format(components.shape)
    standardizer = StandardScaler()
    components = standardizer.fit_transform(components)
    
    # Shuffle factor analysis X rows. (metrics x factors)
    metric_shuffle_indices = get_shuffle_indices(components.shape[0])
    components = components[metric_shuffle_indices]
    
    # Make labels for each column.
    components_columnlabels = X.columnlabels[metric_shuffle_indices]
    
    return components, components_columnlabels



def run_factor_analysis(X):
    """Execute factor analysis.
    Arg
      X : X data with column, row label. (Matrix)
    
    Return
      components : Result of factor analysis in variance descending order. 
                   (Numpy array, [# of features, # of components]) 
      components_columnlabels : Labels for each componenets. (Numpy array, [# of features,]) 
    """   
    
    
    #-------------------------      
    # Execute factor analysis
    #-------------------------
    
    fa = FactorAnalysis()
    # Feed X.data.T for reduction across feature axis, X.data for reduction across sample axis.
    fa.fit(X.data)


    #-----------------------------
    # Determine number of factors
    #-----------------------------
    
    # Only nonzero components should be considered.
    fa_mask = np.sum(fa.components_ != 0.0, axis=1) > 0.0

    # Calculate each variance(actually sum of absoulute value) and total variance
    variances = np.sum(np.abs(fa.components_[fa_mask]), axis=1)
    total_variance = np.sum(variances).squeeze()
    print "total variance: {}".format(total_variance)
    
    # Select number of components which can explain REQUIRED_VARIANCE_EXPLAINED percent of variance.
    var_exp = np.array([np.sum(variances[:i+1]) / total_variance * 100 
                        for i in range(variances.shape[0])])
    factor_cutoff = np.count_nonzero(var_exp < REQUIRED_VARIANCE_EXPLAINED) + 1
    factor_cutoff = min(factor_cutoff, 10)
    print "factor cutoff: {}".format(factor_cutoff)
    for i,var in enumerate(variances):
        print i, var, np.sum(variances[:i+1]), np.sum(variances[:i+1]) / total_variance


    #----------------
    # Postprecessing
    #----------------
    
    # Standardization
    components = np.transpose(fa.components_[:factor_cutoff]).copy()
    print "components shape: {}".format(components.shape)
    standardizer = StandardScaler()
    components = standardizer.fit_transform(components)
    
    # Shuffle factor analysis X rows. (metrics x factors)
    metric_shuffle_indices = get_shuffle_indices(components.shape[0])
    components = components[metric_shuffle_indices]
    
    # Make labels for each column.
    components_columnlabels = X.columnlabels[metric_shuffle_indices] 
    
    return (components, components_columnlabels)

def run_hierarchy(components, components_columnlabels, savedir='./results', cluster_range=[4]):
    hier = Hierarchy_clustering_(components, cluster_range)
    plt.scatter(components[:, 0], components[:, 1], color=colors[hier.cluster_map_[0]].tolist(), s=10)
    plt.show()  

def run_spectral(components, component_columnlabels, savedir='./results', cluster_range=[10]):
    sp = Spectral_clustering_(components, cluster_range)
    plt.scatter(components[:, 0], components[:, 1], color=colors[sp.cluster_map_[0]].tolist(), s=10)
    plt.show()  

    
def run_mog(components, component_columnlabels, savedir='./results', cluster_range=[10]):
    #plt.scatter(components[:,0], components[:, 1])
    #plt.show()
    mog = GaussianMixture_(components, cluster_range)
    print mog.cluster_map_[0]
    plt.scatter(components[:, 0], components[:, 1], color=colors[mog.cluster_map_[0]].tolist(), s=10)
    plt.show() 

def run_kmeans(components,component_columnlabels,savedir='./results',cluster_range=[10],
               algorithms=["DetK","GapStatistic"]):    
    
    #----------------------------------------------------------------------------------------
    # Execute k-means algorithm on various k value and plot its within sum of square over K. 
    #----------------------------------------------------------------------------------------
    
    kmeans = KMeans_(components, cluster_range)
    kmeans.plot_results(savedir, components, component_columnlabels)
    

    #------------------------------------------
    # Determine  optimal number of clusters K.
    #------------------------------------------
    
    for algorithm in algorithms:
        kselection = KSelection.new(components, cluster_range,
                                        kmeans.cluster_map_, algorithm)
        print "{} optimal # of clusters: {}".format(algorithm,
                                                    kselection.optimal_num_clusters_)
        kselection.plot_results(savedir)
    
    
    #---------------------------------------------------
    # Select representative datapoint for each cluster.
    #---------------------------------------------------
    
    metric_clusters = {}
    featured_metrics = {}
    
    # for loop for changing k value.
    for n_clusters, (cluster_centers,labels,_) in kmeans.cluster_map_.iteritems():
        # n_clusters : k value.
        # cluster_centers : Center of each clusters.
        # labels : Label for each datapoint's cluster assignment result.

        # For each cluster, calculate the distances of each metric from the
        # cluster center. We use the metric closest to the cluster center.
        mclusters = []
        mfeat_list = []
        for i in range(n_clusters):
            metric_labels = component_columnlabels[labels == i] # Name array for particular cluster.
            component_rows = components[labels == i] # Dimension reduced data array for particular cluster.
            centroid = np.expand_dims(cluster_centers[i], axis=0) # Centroid datapoint.
            
            # Calculate distance between each datapoint and centroid.
            dists = np.empty(component_rows.shape[0]) # Distance array. 
            for j,row in enumerate(component_rows): 
                row = np.expand_dims(row, axis=0)
                dists[j] = cdist(row, centroid, 'euclidean').squeeze()
            
            # Order metric by distance between centroid.
            order_by = np.argsort(dists)
            metric_labels = metric_labels[order_by]
            dists = dists[order_by]
            mclusters.append((i,metric_labels, dists))
            
            assert len(OPT_METRICS) > 0
            label_mask = np.zeros(metric_labels.shape[0])
            for opt_metric in OPT_METRICS:
                label_mask = np.logical_or(label_mask, metric_labels == opt_metric)
            if np.count_nonzero(label_mask) > 0:
                mfeat_list.extend(metric_labels[label_mask].tolist())
            elif len(metric_labels) > 0:
                mfeat_list.append(metric_labels[0])
        metric_clusters[n_clusters] = mclusters
        featured_metrics[n_clusters] = mfeat_list

    # Print out featured metrics.
    for n_clusters, mlist in sorted(featured_metrics.iteritems()):
        savepath = os.path.join(savedir, "featured_metrics_{}.txt".format(n_clusters))
        with open(savepath, "w") as f:
            f.write("\n".join(str(sorted(mlist))))
    
    # Print out membership information. 
    for n_clusters, memberships in sorted(metric_clusters.iteritems()):
        cstr = ""
        for i,(cnum, lab, dist) in enumerate(memberships):
            assert i == cnum
            cstr += "---------------------------------------------\n"
            cstr += "CLUSTERS {}\n".format(i)
            cstr += "---------------------------------------------\n\n"
             
            for l,d in zip(lab,dist):
                cstr += "{}\t({})\n".format(l,d)
            cstr += "\n\n"
 
        savepath = os.path.join(savedir, "membership_{}.txt".format(n_clusters))
        with open(savepath, "w") as f:
            f.write(cstr)

if __name__ == "__main__":
    main()
