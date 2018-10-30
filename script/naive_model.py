from __future__ import print_function


import numpy as np
import itertools
from sklearn.neighbors import BallTree

from gen_data import *
from get_s import plot_location_list
from scipy import spatial
from sklearn.metrics import *
import pickle
import feature_extraction as fe
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run naive_model.py")
    # parser = argparse.ArgumentParser(description="Run naive_model.py")

    parser.add_argument('--KFold', type=int, default=3, help='Number of Fold. Default is 3.')
#
    parser.add_argument('--embedding', type=int, default=0, help='Feature hyperparameter. Default is 0.')

    return parser.parse_args()

def process_locs(train_locs, validation_locs, test_locs):
    # print(train_locs)
    all_locs = list(itertools.chain(train_locs, validation_locs, test_locs))
    X = []
    for loc in all_locs:
        # print(loc.split(' ')[0],loc.split(' ')[2])
        X.append([float(loc.split(' ')[0]),-float(loc.split(' ')[2])])
    return np.array(X)


### generate 3 different plots to see randomness
# plot_location_list(train_locs, validation_locs, test_locs)

'''
def weighted_avg_old(train, validation, train_locs, validation_locs):
    val_predict = []
    for val in range(len(validation_locs)):
        print(val, validation_locs[val])
        X = list(itertools.chain(train_locs, [validation_locs[val]]))
        tree = BallTree(X, leaf_size=2, metric='haversine')
        dist, ind = tree.query([validation_locs[val]], k=5)
        X = np.array(X)
        print(X[ind[0, 1:]])
        plot_location_list([X[-1]], X[ind[0, 1:]], X[ind[0, 1:]])
        # print(ind[0, 1:])
        # avg_vec = [x * y for x, y in zip(train[ind[0, 1:]], dist[ind[0, 1:]])]
        avg_vec = np.zeros(len(train[0]))
        for x, y in zip(train[ind[0, 1:]], dist[0,1:]):
            avg_vec+=x*y
        # print(avg_vec)
        avg_vec = [1 if x > 0 else 0 for x in avg_vec]
        val_predict.append(avg_vec)
        # break
    val_predict = np.array(val_predict)
    print(len(validation),len(val_predict))

    for y, y_pred in zip(validation.T, val_predict.T):
        result = 1 - spatial.distance.cosine(y, y_pred)
        print(result)
        # print(f1_score(y, y_pred))

def fixed_split():
    train, validation, test, train_locs, validation_locs, test_locs  = split_data(10, True)

    # baselines
    line_vecs_train = fe.extract_subway_line_vec(train_locs)
    line_vecs_validation = fe.extract_subway_line_vec(validation_locs)

    weighted_avg(train, validation, train_locs, validation_locs)
    onevsrest(line_vecs_train, train, line_vecs_validation, validation)
    rf(line_vecs_train, train, line_vecs_validation, validation)
    nn(line_vecs_train, train, line_vecs_validation, validation)
'''
def accuracy_instance(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in xrange(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def evaluation(y, y_predict):
    # print ("p, r, F1 micro, hamming loss")
    print (precision_score(y, y_predict,average='micro'),
           recall_score(y, y_predict,average='micro'),
           f1_score(y, y_predict, average='micro'),
           accuracy_instance(y, y_predict),
           hamming_loss(y, y_predict), sep="\t")
    # print ("accuracy", accuracy_score(y, y_predict))
    # print ("hamming loss", hamming_loss(y, y_predict))

def weighted_avg(train, validation, train_locs, validation_locs):
    val_predict = []
    tree = BallTree(train_locs, leaf_size=2, metric='haversine')

    for val in range(len(validation_locs)):
        # print(val, validation_locs[val])
        dist, ind = tree.query([validation_locs[val]], k=5)
        # print (validation_locs[val], train_locs[ind])
        # plot_location_list([validation_locs[val]], train_locs[ind][0], train_locs[ind][0])

        # Use of Inverse distance weight here
        idw=(1/dist)/((1/dist).sum())
        # print (train[ind][0].shape, idw.shape)
        r = np.dot(idw, train[ind][0]).T
        r = np.squeeze(r)
        # print(r.shape, train.shape)

        r = [1 if x > 0 else 0 for x in r]
        # print (r)
        val_predict.append(r)
        # break
    val_predict = np.array(val_predict)
    # print(validation.shape, val_predict.shape)

    # print("Weighted avg")
    evaluation(validation, val_predict)


def onevsrest(X, y, X_v, y_v):
    # clf = OneVsRestClassifier(LinearSVC(kernal='sigmoid', random_state=0))
    clf = OneVsRestClassifier(SVC(C = 5.0,random_state=0))
    y_predict = clf.fit(X, y).predict(X_v)

    # print (y_predict.shape)
    #print("OneVsRestClassifier")
    # print("OneVsRestClassifier LinearSVC")
    # print(clf.estimator)
    evaluation(y_v, y_predict)

    # check label distribution
    # print(y.sum(axis=0))
    # print(y.mean(axis=1).max())



def rf(line_vecs_train, train, line_vecs_validation, validation):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(line_vecs_train, train)
    y_predict = clf.predict(line_vecs_validation)
    # print("Random Forest")
    # print(clf.estimators_)
    evaluation(validation, y_predict)


def nn(line_vecs_train, train, line_vecs_validation, validation):
    #default: clf = MLPClassifier(hidden_layer_sizes=(30, 30))
    # good
    clf = MLPClassifier(hidden_layer_sizes=(32, 32))#, activation = 'logistic')
    clf.fit(line_vecs_train, train)
    # see training acc
    # y_predict_train = clf.predict(line_vecs_train)
    # evaluation(train, y_predict_train)

    y_predict = clf.predict(line_vecs_validation)
    # print("MLP")
    # print(clf.estimator)
    evaluation(validation, y_predict)

'''
K: number of folds. Will select 1 fold for testing, others for training,
CURRENT UPDATE: ALL LABELS/SPECIES NOT IN TRAINING ARE REMOVED FROM BOTH
'''
def cv(args):
    K = args.KFold
    k_folds_X, k_folds_loc = K_fold_split(K, 20, True)
    for test_ind in range(K):
        print("In Fold", test_ind)

        test_X = k_folds_X[test_ind]
        test_locs = k_folds_loc[test_ind]
        train_X = []
        train_locs = []
        for train_ind in range(K):
            if train_ind == test_ind:
                continue
            train_X+=list(k_folds_X[train_ind])
            train_locs+=list(k_folds_loc[train_ind])

        train_Y = np.array(train_X)
        train_locs = np.array(train_locs)
        test_Y = np.array(test_X)
        test_locs = np.array(test_locs)

        # '''
        # get labels not in training set, and remove them in both training and test

        empty_labels = np.where(np.sum(train_Y, axis=0) == 0)[0]
        # print(len(empty_labels))
        train_Y = np.delete(train_Y, empty_labels, 1)
        test_Y = np.delete(test_Y, empty_labels, 1)
        # print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
        # '''

        # print(np.where(train_X[:, 612] > 0))
        # line_vecs_train = fe.extract_subway_line_vec(train_locs)
        # line_vecs_test = fe.extract_subway_line_vec(test_locs)

        # print(train_locs)

        line_vecs_train = fe.extract_feats(train_locs, args.embedding)
        line_vecs_test = fe.extract_feats(test_locs, args.embedding)
        # print(line_vecs_train[:3])
        # print(line_vecs_train.shape, line_vecs_test.shape)

        print("precision\trecall\tF1 micro\taccuracy\thamming loss")
        # weighted_avg(train_Y, test_Y, train_locs, test_locs)
        # onevsrest(line_vecs_train, train_Y, line_vecs_test, test_Y)
        # rf(line_vecs_train, train_Y, line_vecs_test, test_Y)
        nn(line_vecs_train, train_Y, line_vecs_test, test_Y)
        print("------------------------------")
        # break



warnings.filterwarnings("ignore")


if __name__ == '__main__':
    args = parse_args()
    if args.embedding == 2:
        print("Station + embedding features")
    elif args.embedding == 1:
        print("Only embedding features")
    elif args.embedding == 0:
        print("Only station features")
    else:
        print("Invalid, use default 2")

    cv(args)


# uncomment to plot the data
# plot_location_list([X[0]], X[ind[0, 1:]], X[ind[0, 1:]])

