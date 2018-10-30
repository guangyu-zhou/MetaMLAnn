from __future__ import print_function

import numpy as np
import pandas as pd
# import gmplot
import pickle
from gmplot import GoogleMapPlotter as gmp
from geopy.distance import vincenty
from scipy import spatial
from sklearn.metrics import jaccard_similarity_score
import matplotlib.pyplot as plt
import pylab
from itertools import combinations

# return a dictionary. key: species name, value: ratio
def extract_species(in_file):
    df1 = np.loadtxt(in_file, skiprows = 2, dtype = 'str')

    s1 = {}
    # no_species = True
    # num_no_s = 0
    for i in range(len(df1)):
        if "|s__" in df1[i][0] and "|t__" not in df1[i][0]:
            # print(df1[i][0]))
            s1[df1[i][0]] = (float(df1[i][1]) + float(df1[i][2]))/2
            # no_species = False

    # if no_species:
        # print(in_file, "No S")

        # break

    return s1

def save_species_pairs():
    all_species_df = np.loadtxt("../files/all_species_new.txt", dtype='str')
    gen_spec = {}
    spec_check = []
    for i in range(len(all_species_df)):
        vec = all_species_df[i].split('|')
        # print(vec[5], vec[6])
        if vec[5] not in gen_spec:
            gen_spec[vec[5]] = {vec[6]}
        else:
            gen_spec[vec[5]].add(vec[6])

    for k in gen_spec.keys():
        print(k, len(gen_spec[k]))
        for elem in combinations(gen_spec[k], 2):
            # print(elem,)
            spec_check.append(elem)
    return spec_check

def save_species_pairs_all():
    all_species_df = load_df()
    gen_spec = {}
    spec_check = []
    spec_count = {}
    for i in range(len(all_species_df)):
        # print(i)
        for j in range(len(all_species_df[i])):
            if "|s__" in all_species_df[i][j][0]:# and "|t__" not in all_species_df[i][j][0]:
                vec = all_species_df[i][j][0].split('|')
                # print(all_species_df[i][j][0])

                # print(vec[5], vec[6])


                if vec[6] not in spec_count:
                    spec_count[vec[6]] = 0
                else:
                    spec_count[vec[6]]+=1
    # print(len(spec_count))
    for i in range(len(all_species_df)):
        for j in range(len(all_species_df[i])):
            if "|s__" in all_species_df[i][j][0]:# and "|t__" not in all_species_df[i][j][0]:
                vec = all_species_df[i][j][0].split('|')

                # print(vec[5], vec[6])
                # print(spec_count[vec[6]])
                if spec_count[vec[6]] > 3:
                    if vec[5] not in gen_spec:
                        gen_spec[vec[5]] = {vec[6]}
                    else:
                        gen_spec[vec[5]].add(vec[6])
                # else:
                #     print("passing", vec[6])


    for k in gen_spec.keys():
        # print(k, len(gen_spec[k]))
        for elem in combinations(gen_spec[k], 2):
        # for elem in combinations(gen_spec.values(), 2):
        #     print(elem,)
            spec_check.append(elem)
    print (len(spec_check))
    return spec_check


def check_similar_species(df1, elem):
    flag1 = False
    flag2 = False
    for i in range(len(df1)):
        # print(df1[i])
        if str(elem[0]) in df1[i][0]:
            flag1 = True
        if str(elem[1]) in df1[i][0]:
            flag2 = True
    if flag1 and flag2:
        return 3
    if flag1 and (not flag2):
        return 1
    if (not flag1) and flag2:
        return 2
    if not flag1 and not flag2:
        return 0


    # return flag1 and flag2



def load_meta():
    in_file = '../files/CAMDA_MetaSUB_NY.txt'
    # df_meta = np.loadtxt(in_file, skiprows = 1, dtype = 'str', delimiter = '\t')
    df = pd.read_csv(in_file, delimiter = '\t')
    df = df.sort_values(['Run_s'], ascending=[True])
    df_meta = df[['Run_s','lat_lon_s']]
    dict = df_meta.set_index('Run_s').to_dict()
    # print(dict)
    return dict['lat_lon_s']

def geo_dist(lat1, lon1, lat2, lon2):

    loc1 = (float(lat1), -float(lon1))
    loc2 = (float(lat2), -float(lon2))
    # print(loc1, loc2)

    distance = vincenty(loc1, loc2).miles
    # print(distance)
    return distance

def cos_sim(vec1, vec2):
    # print(vec1, vec2)
    result = 1 - spatial.distance.cosine(vec1, vec2)
    # print(result)
    return result



def agg_location():
    in_file = '../files/CAMDA_MetaSUB_NY.txt'
    df = pd.read_csv(in_file, delimiter = '\t')
    df = df.sort_values(['Run_s'], ascending=[True])
    df_meta = df[['Run_s','lat_lon_s']]
    res = {k: list(v) for k,v in df_meta.groupby("lat_lon_s")["Run_s"]}

    return res

def extract_vec_by_loc():
    extracted_set = {}
    all_species = set()
    sample_vec = {}
    for i in range(1748535, 1750107,1):
        if i == 1749159 or i == 1750038:
                continue
        s1 = extract_species("../profile_merged/profiled_SRR" + str(i) +".txt")
        all_species |= set(s1.keys())
        extracted_set['SRR'+str(i)] = s1

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
            if(len(_spec.keys())) < 3:
                # print(sample, len(_spec.keys()))
                continue

            for elem in all_species:
                # if 'Aspergillus' in elem:
                #     print(elem)
                # print(_spec.keys())
                if elem in _spec.keys():
                    result.append(_spec[elem]/100.0)
                else:
                    result.append(0.0)
            result = np.array(result)
            loc_species[loc].append(result)

        if len(loc_species[loc]) == 0:
            print("Empty location!")
        # print(np.mean(loc_species[loc], axis = 0))
        #assert(len(loc_species[loc]) > 0)
        #if len(loc_species[loc]) == 0:
        #	del loc_species[loc]
        #	continue
        #print(loc_species[loc])
        loc_species[loc] = np.mean(loc_species[loc], axis = 0)
        loc_species[loc] = [ 1 if x > spec_thre else 0 for x in loc_species[loc]]


        # if sum(loc_species[loc]) < 3:
            # del loc_species[loc]
            # continue
        vec_list.append(loc_species[loc])
        loc_valid.append(loc)

    print("Number of locations",len(loc_valid), len(vec_list), "number of species(after filter) ", len(vec_list[0]), len(all_species))
    # plt.hist(key_len_plot,  bins=range(min(key_len_plot), max(key_len_plot) + 3, 3))
    # plt.show()

    # print(vec_list)
    with open("../files/all_species_new.txt", "w") as output:
        for elem in all_species:
            output.write(elem+'\n')

    return vec_list, loc_valid
    # ================================================================================

def plot_by_loc(vec_list, loc_sample_list, log_form):

    # loc_sample_list = agg_location().keys()
    sim_mat = compute_pairwise_sim(vec_list)


    x = []
    y = []
    for i in range(len(loc_sample_list) - 1):
        for j in range(i+1, len(loc_sample_list)):
            # print(i,j, loc_sample_list[i], loc_sample_list[j])
            # print(res[samples[i]], res[samples[j]])
            dist = geo_dist(loc_sample_list[i].split(' ')[0],loc_sample_list[i].split(' ')[2], loc_sample_list[j].split(' ')[0],loc_sample_list[j].split(' ')[2])

            if dist > 30:
                # print("Outlier, skip", loc_sample_list[i], loc_sample_list[j])
                continue
            sim = sim_mat[i][j]
            ## sim = jaccard_similarity_score(vec_list[i], vec_list[j])
            # sim = jaccard(vec_list[i], vec_list[j], len(vec_list[0]))
            # inter = 0
            # union = 0
            # for idx in range(len(vec_list)):
            # 	if vec_list[i][idx] == vec_list[j][idx]  and vec_list[i][idx] == 1:
            # 		inter+=1
            # 		union+=1
            # 	elif vec_list[i][idx] == 1 or vec_list[j][idx] == 1:
            # 		union+=1
            # print(inter, union)


            # print(i,j,sim)
            if log_form:
                x.append(np.log2(dist))
                y.append(np.log2(sim))
            else:
                x.append(dist)
                y.append(sim)
            # print(res[sample].split(' ')[0],res[sample].split(' ')[2], vec[sample][:100])
        # break


    # from scipy.optimize import curve_fit
    # x.sort()
    # plt.plot(x, 'o')
    # plt.show()

    # print(len(x), len(y))
    # print(x)
    # print(y)
    plt.scatter(x, y, label = 'data', s = 0.5)
    plt.xlabel('dist')
    plt.ylabel('sim')
    plt.legend()
    plt.show()

def compute_pairwise_sim(mat):
    # from sklearn.metrics.pairwise import cosine_similarity
    # from scipy import sparse
    # similarities = cosine_similarity(sparse.csr_matrix(np.array(mat)))
    # return similarities

    from sklearn.metrics.pairwise import pairwise_distances
    # D = pairwise_distances(mat, metric='jaccard')
    D = pairwise_distances(mat, metric='cosine')
    return D

def run_PCA():
    from sklearn import decomposition
    X, locs = extract_vec_by_loc()
    # print(X)
    pca = decomposition.PCA(n_components=64)
    pca.fit(X)
    X = pca.transform(X)
    print("PCA sum",np.sum(pca.explained_variance_ratio_))
    # print(len(X), len(X[0]))
    # print(X)
    return X, locs

def jaccard(x, y, nfeats):
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    # n11 = sum((c1==1)&(c2==1))
    # n00 = sum((c1==0)&(c2==0))
    # jac = float(n11) / (nfeats-n00)
    # return jac
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

def compute_dominant(X):
    mat = np.array(X)
    ratio = []
    for col in range(len(X[0])):
        if np.sum(mat[:,col]) > 20:
            print(col, np.sum(mat[:,col]), float(np.sum(mat[:,col]))/len(X[0]))
            ratio.append(float(np.sum(mat[:,col]))/len(X[0]))

    plt.hist(ratio)#,  bins=range(min(key_len_plot), max(key_len_plot) + 3, 3))
    plt.show()



# X, locs = run_PCA()
spec_thre = 0.
# X, locs = extract_vec_by_loc()
# for row in locs:
    # print(row)
# for row in X:
# 	print(np.sum(row))
# X = np.array(X)
# print (X[:,612])
# print(np.where(X[:,612]>0))
# np.savetxt('../files/data_binary_no_filter.txt',X)
# compute_dominant(X)
# plot_by_loc(X, locs,log_form = False)
spec_check = [
    ["s__Bacillus_licheniformis", "s__Bacillus_amyloliquefaciens"],
    ["s__Pseudomonas_sp_HPB0071", "s__Pseudomonas_mendocina"],
    ["s__Alishewanella_aestuarii", "s__Alishewanella_jeotgali"],
    ["s__Clostridium_beijerinckii", "s__Clostridium_perfringens"],
    ["s__Corynebacterium_callunae", "s__Corynebacterium_casei"],
    ["s__Acinetobacter_johnsonii", "s__Acinetobacter_pittii_calcoaceticus_nosocomialis"],
    ["s__Escherichia_coli", "s__Acinetobacter_pittii_calcoaceticus_nosocomialis"]
]

# spec_check = []
# for i in range(1748535, 1750107, 1):
#     if i == 1749159 or i == 1750038:
#         continue
#     # check_similar_species("../profile_merged/profiled_SRR" + str(i) + ".txt")

#     for k in gen_spec.keys():
#         # print(gen_spec[k])
#         for elem in combinations(gen_spec[k],2):
#             # print(elem,)
#             spec_check.append(elem)

    # break



# pickle.dump(spec_check, open("./tmp/spec_check.p", 'w'))

# spec_check= pickle.load(open("./tmp/spec_check.p", 'r'))
# df1 = np.loadtxt(in_file, skiprows=2, dtype='str')

def load_df():
    dfs = []
    for i in range(1748535, 1750107, 1):
        if i == 1749159 or i == 1750038:
            continue
        dfs.append(np.loadtxt("../profile_merged/profiled_SRR" + str(i) + ".txt", skiprows=2, dtype='str'))
    return dfs

dfs = load_df()

spec_check = [
    ["s__Bacillus_licheniformis", "s__Bacillus_amyloliquefaciens"],
    ["s__Pseudomonas_sp_HPB0071", "s__Pseudomonas_mendocina"],
    ["s__Alishewanella_aestuarii", "s__Alishewanella_jeotgali"],
    ["s__Clostridium_beijerinckii", "s__Clostridium_perfringens"],
    ["s__Corynebacterium_callunae", "s__Corynebacterium_casei"],
    ["s__Acinetobacter_johnsonii", "s__Acinetobacter_pittii_calcoaceticus_nosocomialis"],
    ["s__Escherichia_coli", "s__Acinetobacter_pittii_calcoaceticus_nosocomialis"]
]
spec_check = save_species_pairs_all() # remove small species also
# spec_check = save_species_pairs()
# print("spec_check",len(spec_check))
#
print("Appear none, 1st species only, 2nd species only, both")
for i in range(len(spec_check)):
    t0 = 0
    t1 = 0
    t2 = 0
    t3 = 0
    elem = spec_check[i]
    for df in dfs:
        if check_similar_species(df,elem) == 3:
            t3+=1
        if check_similar_species(df,elem) == 2:
            t2+=1
        if check_similar_species(df, elem) == 1:
            t1+=1
        if check_similar_species(df, elem) == 0:
            t0+=1
    print(i, "tot for:",elem,t0,t1,t2,t3)
    # break
# print(spec_check)

