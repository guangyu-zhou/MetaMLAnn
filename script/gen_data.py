from __future__ import print_function

import numpy as np
import pandas as pd
import random

# import gmplot
import pickle
from gmplot import GoogleMapPlotter as gmp
from geopy.distance import vincenty
from scipy import spatial
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt
import pylab


# return a dictionary. key: species name, value: ratio
def extract_species(in_file):
    df1 = np.loadtxt(in_file, skiprows=2, dtype='str')

    s1 = {}
    no_species = True
    num_no_s = 0
    for i in range(len(df1)):
        if "|s__" in df1[i][0] and "|t__" not in df1[i][0]:
            # print(df1[i][0])
            s1[df1[i][0]] = (float(df1[i][1]) + float(df1[i][2])) / 2

    return s1


def geo_dist(lat1, lon1, lat2, lon2):
    loc1 = (float(lat1), -float(lon1))
    loc2 = (float(lat2), -float(lon2))
    # print(loc1, loc2)

    distance = vincenty(loc1, loc2).miles
    # print(distance)
    return distance


# generate dictionary: key: different locations; value: list of sample IDs
def agg_location():
    in_file = '../files/CAMDA_MetaSUB_NY.txt'
    df = pd.read_csv(in_file, delimiter='\t')
    df = df.sort_values(['Run_s'], ascending=[True])
    df_meta = df[['Run_s', 'lat_lon_s']]
    res = {k: list(v) for k, v in df_meta.groupby("lat_lon_s")["Run_s"]}



    return res

'''
key function for generating vectors:
return
1) vec_list: list of dicts. Each dicts, key: location, value: a list of species abundances (ordered)
The ordered list of species names are stored as a file.
2) loc_valid: list of all locations (ordered).

'''
def extract_vec_by_loc(spec_thre):
    extracted_set = {}
    all_species = set()
    sample_vec = {}
    for i in range(1748535, 1750107, 1):
        if i == 1749159 or i == 1750038:
            continue
        s1 = extract_species("../profile_merged/profiled_SRR" + str(i) + ".txt")
        all_species |= set(s1.keys())
        extracted_set['SRR' + str(i)] = s1

    all_species = list(all_species)
    loc_samples = agg_location()

    loc_species = {}
    vec_list = []
    loc_sample_list = loc_samples.keys()

    loc_valid = []

    # key_len_plot = []

    for loc in loc_sample_list:
        if loc == 'not applicable':
            continue
        loc_number = (float(loc.split(' ')[0]), -float(loc.split(' ')[2]))
        if loc_number[0] > 41.0 or loc_number[0] < 40.0 or loc_number[1]< -75.0 or loc_number[1] > -73.0:
            continue
        loc_species[loc] = []
        for sample in loc_samples[loc]:
            if sample == 'SRR1749159' or sample == 'SRR1750038':
                continue
            # print(len(extracted_set[sample]),len(loc_species[loc] ))
            result = []
            _spec = extracted_set[sample]
            # print("Key length", len(_spec.keys()))
            # key_len_plot.append(len(_spec.keys()))
            ''' remove those species less than 3'''
            if (len(_spec.keys())) < 3:
                # print(sample, len(_spec.keys()))
                continue

            for elem in all_species:
                if elem in _spec.keys():
                    result.append(_spec[elem] / 100)
                else:
                    result.append(0.0)
            result = np.array(result)
            loc_species[loc].append(result)

        if len(loc_species[loc]) == 0:
            print("Empty location!")
        # print(np.mean(loc_species[loc], axis = 0))
        # assert(len(loc_species[loc]) > 0)
        # if len(loc_species[loc]) == 0:
        #	del loc_species[loc]
        #	continue
        # print(loc_species[loc])
        loc_species[loc] = np.mean(loc_species[loc], axis=0)
        loc_species[loc] = [1 if x > spec_thre else 0 for x in loc_species[loc]]

        # if sum(loc_species[loc]) < 3:
        # del loc_species[loc]
        # continue
        vec_list.append(loc_species[loc])
        loc_valid.append(loc_number)

    print("Number of locations", len(loc_valid), len(vec_list), "number of species(after filter) ", len(vec_list[0]))
    with open("../files/all_species.txt", "w") as output:
        for elem in all_species:
            output.write(elem + '\n')
    X = np.array(vec_list)
    # print(np.where(X[:, 612] > 0))
    np.savetxt('../files/data_binary_no_filter.txt', X)

    return vec_list, loc_valid


def split_data(seed_no, pickle_files = False):
    # fix 14 here: good mixture
    np.random.seed(seed_no)

    if pickle_files:
        print('loading from file')
        X = pickle.load(open("./tmp/pic_X.data", 'r'))
        locs = pickle.load(open("./tmp/pic_locs.data", 'r'))
    else:
        X, locs = extract_vec_by_loc(spec_thre = 0.0)
        for i in range(len(locs)):
            locs[i] = (float(locs[i].split(' ')[0]), -float(locs[i].split(' ')[2]))
        pickle.dump(X, open("./tmp/pic_X.data", 'w'))
        pickle.dump(locs, open("./tmp/pic_locs.data", 'w'))
    locs = np.array(locs)
    X = np.array(X)

    indices = np.arange(len(locs))
    np.random.shuffle(indices)
    train_ind = indices[:len(locs)*3 / 5]
    validation_ind = indices[len(locs)*3 / 5:len(locs)*4 / 5]
    test_ind = indices[len(locs)*4 / 5:]
    # print(train_ind, validation_ind, test_ind)



    train =X[train_ind,:]
    validation = X[validation_ind,:]
    test = X[test_ind,:]

    print("Number of training, validation, test:", len(train), len(validation),len(test))
    # print(validation_ind, test_ind)

    return train, validation, test, locs[train_ind], locs[validation_ind], locs[test_ind]


'''
Randomly split the dataset into K folds. Includes the species matrix (Y) and the location list (loc)

Return: 2 list of each contains K Folds
'''
def K_fold_split(K, seed_no = 14, pickle_files = False):
    # fix 14 here: good mixture
    np.random.seed(seed_no)

    if pickle_files:
        print('loading from file')
        Y = pickle.load(open("./tmp/pic_spec_vec.data", 'r'))
        locs = pickle.load(open("./tmp/pic_locs.data", 'r'))

    else:
        Y, locs = extract_vec_by_loc(spec_thre = 0.0)

        # print(locs)
        # for i in range(len(locs)):
            # print(locs[i], float(locs[i].split(' ')[0]), -float(locs[i].split(' ')[2]))
            # temp = (float(locs[i].split(' ')[0]), -float(locs[i].split(' ')[2]))
            # locs[i] = temp
        pickle.dump(Y, open("./tmp/pic_spec_vec.data", 'w'))
        pickle.dump(locs, open("./tmp/pic_locs.data", 'w'))

    Y = np.array(Y)
    locs = np.array(locs)
    indices = np.arange(len(locs))
    np.random.shuffle(indices)

    indices = np.array_split(np.array(indices), K)
    # for elem in indices:
    #     print(elem)
    k_folds_Y = [Y[indices[i],:] for i in range(K)]
    # k_folds_loc = []
    # for i in range(K):
    #     one_fold_loc = []
    #     for idx in indices[i]:
    #         one_fold_loc.append(tuple(locs[idx]))
    #     k_folds_loc.append(one_fold_loc)
    k_folds_loc = [ locs[indices[i],:] for i in range(K)]
    # print(k_folds_loc[0], k_folds_loc[1])
    # print(len(X[indices[0],:][0]))
    # print(len(k_folds_X), len(k_folds_loc[0]))

    return k_folds_Y, k_folds_loc



def main():
    X = pickle.load(open("./tmp/pic_spec_vec.data", 'r'))
    locs = pickle.load(open("./tmp/pic_locs.data", 'r'))
    # print (locs)
    X = np.array(X)
    # print (X.sum(axis = 1))

if __name__ == '__main__':
    K_fold_split(5, 14, False)
    # main()