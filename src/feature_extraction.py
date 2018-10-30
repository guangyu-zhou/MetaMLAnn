import json
import numpy as np
from pprint import pprint
from sklearn.neighbors import BallTree
import pickle
from sklearn.neighbors import DistanceMetric
from get_s import plot_location_node_processed

'''
return 2 station vectors. one is lat_lon, the other is meta infor.
'''
def load_stations():
    station_coordinates = []
    station_info = []

    with open('../mta_data/subway-stations.geojson') as data_file:
        stations= json.load(data_file)

    # pprint(stations['features'][0])

    for stat in stations['features']:
        lat_lon = (stat['geometry']['coordinates'][1], stat['geometry']['coordinates'][0])
        station_coordinates.append(lat_lon)
        properties = stat['properties']

        station_info.append(properties)
    # print len(station_coordinates),station_coordinates[:2]
    return np.array(station_coordinates), station_info

def extract_subway_line_vec(locs):
    # X = pickle.load(open("./tmp/pic_X.data", 'r'))
    # locs = pickle.load(open("./tmp/pic_locs.data", 'r'))

    station_coordinates, station_info = load_stations()
    # print (station_coordinates[0])
    station_tree = BallTree(station_coordinates, leaf_size=2, metric='haversine')



    # line_label_inverse = {} # dict:label -> index
    line_label_inverse = {u'3': 9, u'1': 22, u'7 Express': 20, u'2': 3, u'5': 4, u'4': 2, u'7': 11, u'6': 10, u'A': 17, u'6 Express': 23, u'C': 21, u'B': 0, u'E': 12, u'D': 1, u'G': 8, u'F': 7, u'J': 14, u'M': 13, u'L': 16, u'N': 5, u'Q': 19, u'S': 18, u'R': 6, u'W': 24, u'Z': 15}

    no_line_locs = []
    line_vecs = []
    for loc in range(len(locs)):

        #initialize the label vector
        line_label_vec = np.zeros(len(line_label_inverse))

        ind, dist  = station_tree.query_radius([locs[loc]], r = 0.01, count_only = False, return_distance = True)

        # deal with non station locsspecially
        if len(dist[0])==0:
            dist, ind2  = station_tree.query([locs[loc]], k = 2)
            # print(ind2)
            label0 = set(station_info[ind2[0][0]]['line'].split('-'))
            label1 = set(station_info[ind2[0][1]]['line'].split('-'))
            label_intersect = label0.intersection(label1)
            for x in label_intersect:
                line_label_vec[line_label_inverse[x]] = 1
            # print(label0, label1, label_intersect)
            if len(label_intersect) == 0:
                no_line_locs.append(locs[loc])

        for idx in ind[0]:
            # print ("NB: ")
            # print station_coordinates[idx], station_info[idx]['name'], station_info[idx]['line'].split('-')
            # print(station_info[idx]['line'].split('-'))
            for x in station_info[idx]['line'].split('-'):
                line_label_vec[line_label_inverse[x]] = 1
                # if x not in line_label_inverse:
                #     line_label_inverse[x] = len(line_label_inverse)


        line_vecs.append(line_label_vec)

    return np.array(line_vecs)

def load_emb(locs):
    locs_all = pickle.load(open("./tmp/pic_locs.data", 'r'))
    emb_file = "../emb/emb16.txt"
    # print locs_all

    locs_emb = {}
    with open(emb_file,'r') as fin:
        emb_vec = np.loadtxt(fin, skiprows=1)#, usecols = range(1,16))
        # print len(emb_vec),len(emb_vec[0])
        for row in emb_vec:
            # print(int(row[0]), row[1:])
            locs_emb[tuple(locs_all[int(row[0])])]= (row[1:])
    # print len(locs_emb)

    emb_vecs = []
    for loc in locs:
        emb_vecs.append(locs_emb[tuple(loc)])
        # print loc, locs_emb[loc]
    return emb_vecs
    #
def construct_network():
    locs = pickle.load(open("./tmp/pic_locs.data", 'r'))
    fout = open("../files/dist_edge.txt", 'w+')
    loc2index = {}

    # print(loc2index)
    dist = DistanceMetric.get_metric('haversine')
    dis_pairwise = dist.pairwise(locs)
    for i in range(len(locs)):
        loc2index[locs[i]] = i
        for j in range(i, len(locs)):
            if i == j:
                continue
            fout.write((str(i) + ' ' + str(j) + ' '+ str(dis_pairwise[i,j]) + '\n'))

def extract_feats(locs, embedding = 2):
    feats_vec = []
    line_vecs = extract_subway_line_vec(locs)
    emb_vecs = load_emb(locs)

    if embedding == 2:
        for v in range(len(locs)):
            # print np.concatenate([line_vecs[v], emb_vecs[v]])
            feats_vec.append(np.concatenate([line_vecs[v], emb_vecs[v]]))
        # print len(feats_vec[0])
        return np.array(feats_vec)
    elif embedding == 1:
        return np.array(emb_vecs)
    elif embedding == 0:
        return line_vecs
    # print line_vecs,emb_vecs


if __name__ == '__main__':
    # X = pickle.load(open("./tmp/pic_X.data", 'r'))
    locs = pickle.load(open("./tmp/pic_locs.data", 'r'))
    extract_feats(locs[2:4], True)

    # construct_network()
    # line_vecs = extract_subway_line_vec(locs)
    # print (line_vecs)

    # plot_location_node_processed(no_line_locs)




















