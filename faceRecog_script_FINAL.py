"""
=======================================================================
Faces recognition example using PCA, NMF, Fast ICA with SVM classifer
=======================================================================

Author: Yamin Tun (citation below)
Reference: Faces recognition example using eigenfaces and SVMs
http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset::

                     precision    recall  f1-score   support

  Gerhard_Schroeder       0.91      0.75      0.82        28
    Donald_Rumsfeld       0.84      0.82      0.83        33
         Tony_Blair       0.65      0.82      0.73        34
       Colin_Powell       0.78      0.88      0.83        58
      George_W_Bush       0.93      0.86      0.90       129

        avg / total       0.86      0.84      0.85       282



"""
#from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.lda import LDA

#from sklearn.grid_search import GridSearchCV
from matplotlib.colors import Normalize

from sklearn import decomposition
#from sklearn.cluster import MiniBatchKMeans
from numpy.random import RandomState
#from sklearn.ensemble import ExtraTreesClassifier

#from sklearn.grid_search import GridSearchCV
#from matplotlib.colors import Normalize

import numpy as np
print(__doc__)

#%matplotlib inline


def plot_portrait(images, title,h, w, n_row, n_col,nImages):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.5 * n_col, 1.5 * n_row))
    gs1 = gridspec.GridSpec(n_row, n_col)
    gs1.update(wspace=0, hspace=0) # set the spacing between axes. 

    for i in range(nImages):
        plt.subplot(gs1[i])
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
        plt.axis('equal')

    plt.suptitle(title)

# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def main():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    ###############################################################################
    # Download the data, if not already on disk and load it as numpy arrays

    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    ###############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    bush_face1 = np.loadtxt("/Users/ytun/Dropbox/Academic_UMass/Firstyear.2014-15/Spring2014/CS589/Project/Code/bush_face.txt")
    bush_face2 = np.loadtxt("/Users/ytun/Dropbox/Academic_UMass/Firstyear.2014-15/Spring2014/CS589/Project/Code/bush_img_side.txt")
    bush_face3 = np.loadtxt("/Users/ytun/Dropbox/Academic_UMass/Firstyear.2014-15/Spring2014/CS589/Project/Code/bush_img_others.txt")
    X_test=np.array([bush_face1.flatten().tolist(),bush_face2.flatten().tolist(),bush_face3.flatten().tolist()])
    y_test=np.array([3,3,3])

    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    #n_jobs=1
    n_components = 150

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))

    rng = RandomState(0)

    classifier_list=('PCA','PCASp','ICA')
   
    trainT_mat=np.zeros(3)
    testT_mat=np.zeros(3)
    projectT_mat=np.zeros(3)

    I=[1]
    #PCA, PCA sparse, ICA
    for i in range(len(classifier_list)):
        if i==1:
            n_components=20
        else:
            n_components=150

            
        t0 = time()
        print 'i',i
        print 'eigen title',classifier_list[i]
        eigenface_titles=classifier_list[i]
        
        if i==0:
            print ('PCA.....')
            pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)

        elif i==1:
            print ('NMF.....')
            pca=decomposition.NMF(n_components=n_components, init='nndsvda', beta=5.0,
                       tol=5e-3, sparseness='components').fit(X_train)
        elif i==2:
            print ('Fast ICA....')
            pca=decomposition.FastICA(n_components=n_components, whiten=True).fit(X_train)

        projectT_mat[i]=(time() - t0)

        
        ###############################################################################
        
        
        if hasattr(pca, 'cluster_centers_'):
            eigenfaces = pca.cluster_centers_.reshape((7, h, w))

        else:
            eigenfaces = pca.components_.reshape((n_components, h, w))

        ###############################################################################
        print("Projecting the input data on the eigenfaces orthonormal basis")
        
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        #print("done in %0.3fs" % (time() - t0))

        
        
        ###############################################################################
        # Train a SVM classification model

        print("Fitting the classifier to the training set")
        
        C_range=np.array([1e3, 5e3, 1e4, 5e4, 1e5])
        gamma_range=np.array([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1])


        param_grid = {'C': C_range,
                      'gamma': gamma_range, }
        #clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
        clf = GridSearchCV(SVC( class_weight='auto'), param_grid)
        clf = clf.fit(X_train_pca, y_train)
        print("done in %0.3fs" % (time() - t0))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)

        trainT_mat[i]=(time() - t0)
        print("Training done in %0.3fs" % ( trainT_mat[i]))

        ###############################################################################
        # Quantitative evaluation of the model quality on the test set

        print("Predicting people's names on the test set")
        t0 = time()
        y_pred = clf.predict(X_test_pca)

        testT_mat[i]=(time() - t0)
        print("Testing done in %0.3fs" % (testT_mat[i]))


        print(classification_report(y_test, y_pred, target_names=target_names))
        print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


        ###############################################################################
        # Plot the gallery of the most significative eigenfaces

        eigenface_titles = 'eigenfaces'#["eigenface %d" % i for i in range(eigenfaces.shape[0])]
        n_eigenfaces=eigenfaces.shape[0]

        nRow=(int)(np.sqrt(n_eigenfaces))

        plot_portrait(eigenfaces[:20], eigenface_titles, h, w,4,5, 20)

        print '\n\n'
    
##    print 'trainT_mat',trainT_mat
##    print 'testT_mat',testT_mat
##    print 'projectT_mat',projectT_mat
    plt.show()
  

        

    ###############################################################################



    ###############################################################################


if __name__ == '__main__':

    main()
