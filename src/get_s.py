from __future__ import print_function

import numpy as np
import pandas as pd
# import gmplot
import pickle
from gmplot import GoogleMapPlotter as gmp

def extract_species(in_file):
	df1 = np.loadtxt(in_file, skiprows = 2, dtype = 'str')

	s1 = {}
	for i in range(len(df1)):
		if "|s__" in df1[i][0]:
			# print(df1[i][0])
			s1[df1[i][0]] = (float(df1[i][1]) + float(df1[i][2]))/2

	return s1

def pairwise_sim():


	# s2 = extract_species("../profile_merged/profiled_SRR1748536.txt")
	# s3 = extract_species("../profile_merged/profiled_SRR1748537.txt")


	# print set1

	# print set2
	# print len(set1),len(set2)
	# print len(set.intersection(set1, set2))
	fout = open("edge_list_10.txt", "w+")
	edge_list = {}
	extracted_set = {}
	for i in range(1748535, 1750107,1):
		if i == 1749159 or i == 1750038:
				continue
		s1 = extract_species("../profile_merged/profiled_SRR" + str(i) +".txt")
		set1 = set(s1.keys())
		extracted_set[i] = set1
	# print extracted_set.keys()

	for i in range(1748535, 1750106,1):
		if i == 1749159 or i == 1750038:
			continue
		# print(i)
		set1 = extracted_set[i]

		for j in range(i+1, 1750107,1):
			if j == 1749159 or j == 1750038:
				continue
			set2 = extracted_set[j]

			# print len(set1),len(set2)
			# if
			# print 'SRR'+str(i),'SRR'+str(j),len(set.intersection(set1, set2))
			sim = len(set.intersection(set1, set2))
			if sim > 10:
				# edge_list[frozenset(('SRR'+str(i),'SRR'+str(j)))] = sim
				# edge_list[(i,j)] = sim
				# fout.write(i,j,sim)
				print(i,j,sim, file=fout)

		# break
	# print "There are",len(edge_list),"edges."
	# pickle.dump(edge_list, open("edge_list.p", "wb"))
	# return edge_list

def pairwise_sim_agg(loc_samples):
	loc_spec = {}

	fout = open("edge_list_agg.txt", "w+")
	edge_list = {}
	extracted_set = {}
	for i in range(1748535, 1750107,1):
		if i == 1749159 or i == 1750038:
				continue
		s1 = extract_species("../profile_merged/profiled_SRR" + str(i) +".txt")
		set1 = set(s1.keys())
		extracted_set['SRR'+str(i)] = set1

	# print(extracted_set)
	for loc in loc_samples.keys():
		loc_spec[loc] = set()
		for sample in loc_samples[loc]:
			if sample == 'SRR1749159' or sample == 'SRR1750038':
				continue
			# print(len(extracted_set[sample]),len(loc_spec[loc] ))
			loc_spec[loc] |= extracted_set[sample]

		# break
		# print(loc_samples[loc])
	# print(len(loc_spec.keys()))
	for i in range(len(loc_spec.keys())-1):
		for j in range(i+1, len(loc_spec.keys())):
			if i == j:
				continue
			print(loc_spec.keys()[i] + '\t' +loc_spec.keys()[j]+"\t"+str(len(set.intersection(loc_spec[loc_spec.keys()[i]], loc_spec[loc_spec.keys()[j]]))), file=fout)
	# print "There are",len(edge_list),"edges."
	# pickle.dump(edge_list, open("edge_list.p", "wb"))
	# return edge_list

def pairwise_dist_sim_agg(loc_samples):
	loc_spec = {}

	fout = open("edge_list_dist_sim.txt", "w+")
	edge_list = {}
	extracted_set = {}
	for i in range(1748535, 1750107,1):
		if i == 1749159 or i == 1750038:
				continue
		s1 = extract_species("../profile_merged/profiled_SRR" + str(i) +".txt")
		set1 = set(s1.keys())
		extracted_set['SRR'+str(i)] = set1

	# print(extracted_set)
	for loc in loc_samples.keys():
		loc_spec[loc] = set()
		for sample in loc_samples[loc]:
			if sample == 'SRR1749159' or sample == 'SRR1750038':
				continue
			# print(len(extracted_set[sample]),len(loc_spec[loc] ))
			loc_spec[loc] |= extracted_set[sample]

		# break
		# print(loc_samples[loc])
	# print(len(loc_spec.keys()))
	for i in range(len(loc_spec.keys())-1):
		for j in range(i+1, len(loc_spec.keys())):
			if i == j:
				continue
			print(loc_spec.keys()[i] + '\t' +loc_spec.keys()[j]+"\t")
			print(loc_spec[loc_spec.keys()[i]])
	# print "There are",len(edge_list),"edges."
	# pickle.dump(edge_list, open("edge_list.p", "wb"))
	# return edge_list


def load_meta():
	in_file = '../CAMDA_MetaSUB_NY.txt'
	# df_meta = np.loadtxt(in_file, skiprows = 1, dtype = 'str', delimiter = '\t')
	df = pd.read_csv(in_file, delimiter = '\t')
	df = df.sort_values(['Run_s'], ascending=[True])
	df_meta = df[['Run_s','lat_lon_s']]
	dict = df_meta.set_index('Run_s').to_dict()
	# print(dict)
	return dict['lat_lon_s']
	# run_s = df_meta['Run_s'].tolist()
	# lat_lon_l  = df_meta['lat_lon_s'].tolist()
	# sample = lat_lon_l
	# print sample
	# return run_s, lat_lon_l


def map():
	geo_l = load_meta()
	# mymap = gmplot.GoogleMapPlotter(40.68, -73.97, 16)
	mymap = gmp.from_geocode("New York")
	path = [[],[]]
	for e in geo_l:
			if e !=  'not applicable':
				path[0].append(float(e.split(' ')[0]))
				path[1].append(-float(e.split(' ')[2]))
				print(e.split(' ')[0],e.split(' ')[2])
	path = [tuple(path[0]),tuple(path[1])]
	# print path
	mymap.scatter(path[0], path[1], c='r', marker=False, s=60)
	mymap.draw("mymap.html")

## generate edge_list.txt
# pairwise_sim()

# return a dict: lat_lon_s: [list of Run_s]
def agg_location():
	in_file = '../CAMDA_MetaSUB_NY.txt'
	df = pd.read_csv(in_file, delimiter = '\t')
	df = df.sort_values(['Run_s'], ascending=[True])
	df_meta = df[['Run_s','lat_lon_s']]
	res = {k: list(v) for k,v in df_meta.groupby("lat_lon_s")["Run_s"]}

	return res

def plot_sample():
	a= load_meta()


	fin = open("edge_list_10.txt", "r")
	it = 1000
	mymap = gmp.from_geocode("New York")

	for line in fin:
		if it%100 == 0:
			print(it)
		if it < 0:
			break
		it-=1

		s1, s2, sim = line.rstrip().split()
		path = [[],[]]
		geo_l =[a["SRR" + str(s1)], a["SRR" + str(s2)]]
		for e in geo_l:
				if e !=  'not applicable':
					path[0].append(float(e.split(' ')[0]))
					path[1].append(-float(e.split(' ')[2]))
		edge = [tuple(path[0]),tuple(path[1])]
		# print(edge, sim)
		mymap.plot(edge[0], edge[1], "plum", edge_width=int(sim)-10,alpha = 0.1)
		mymap.draw("mymap.html")

def plot_location_node(loc_samples):
	loc_spec = {}
	mymap = gmp.from_geocode("New York")
	edge_list = {}
	extracted_set = {}
	for i in range(1748535, 1750107,1):
		if i == 1749159 or i == 1750038:
				continue
		s1 = extract_species("../profile_merged/profiled_SRR" + str(i) +".txt")
		set1 = set(s1.keys())
		extracted_set['SRR'+str(i)] = set1

	path = [[],[],[]]

	for loc in loc_samples.keys():
		if loc ==  'not applicable':
			continue

		loc_spec[loc] = set()
		for sample in loc_samples[loc]:
			if sample == 'SRR1749159' or sample == 'SRR1750038':
				continue
			# print(len(extracted_set[sample]),len(loc_spec[loc] ))
			loc_spec[loc] |= extracted_set[sample]

		path[0].append(float(loc.split(' ')[0]))
		path[1].append(-float(loc.split(' ')[2]))
		path[2].append(len(loc_spec[loc]))

	edge = [tuple(path[0]),tuple(path[1])]
	# mymap.heatmap(edge[0], edge[1], threshold=5, radius=40)
	mymap.scatter(edge[0], edge[1], c = 'b',s=100, marker=False, alpha=1)
	mymap.draw("mymap.html")

def plot_location_node_processed(locs):
	mymap = gmp.from_geocode("New York")
	path = [[],[]]

	for loc in locs:
		if loc ==  'not applicable':
			continue

		path[0].append(loc[0])
		path[1].append(loc[1])

	edge = [tuple(path[0]),tuple(path[1])]
	# mymap.heatmap(edge[0], edge[1], threshold=5, radius=40)
	mymap.scatter(edge[0], edge[1], c = 'b',s=100, marker=False, alpha=1)
	mymap.draw("node_list.html")

def plot_location_edge():
	fin = open("edge_list_agg.txt", "r")
	mymap = gmp.from_geocode("New York")

	i = 0
	for line in fin:
		print(i)
		i+=1
		if i > 200:
			break
		loc1, loc2, sim = line.rstrip().split('\t')
		# print(loc1, loc2, sim)
		path = [[],[]]

		if loc1!=  'not applicable' and loc2!=  'not applicable':
			path[0].append(float(loc1.split(' ')[0]))
			path[0].append(float(loc2.split(' ')[0]))
			path[1].append(-float(loc1.split(' ')[2]))
			path[1].append(-float(loc2.split(' ')[2]))
			edge = [tuple(path[0]),tuple(path[1])]
			print(edge, int(sim)%10)

		else:
			continue
		mymap.plot(edge[0], edge[1], "plum", edge_width=int(sim)%10,alpha = 1)
		mymap.draw("mymap.html")

def plot_location_list(locs1, locs2, locs3):

	mymap = gmp.from_geocode("New York")
	path = [[], []]

	for loc in locs1:
		if loc == 'not applicable':
			continue

		path[0].append(loc[0])
		path[1].append(loc[1])

	edge = [tuple(path[0]), tuple(path[1])]
	# mymap.heatmap(edge[0], edge[1], threshold=5, radius=40)
	mymap.scatter(edge[0], edge[1], c='g', s=100, marker=False, alpha=1)

	path = [[], []]
	for loc in locs2:
		if loc == 'not applicable':
			continue

		path[0].append(loc[0])
		path[1].append(loc[1])

	edge = [tuple(path[0]), tuple(path[1])]
	# mymap.heatmap(edge[0], edge[1], threshold=5, radius=40)
	mymap.scatter(edge[0], edge[1], c='b', s=100, marker=False, alpha=1)

	path = [[], []]
	for loc in locs3:
		if loc == 'not applicable':
			continue

		path[0].append(loc[0])
		path[1].append(loc[1])

	edge = [tuple(path[0]), tuple(path[1])]
	# mymap.heatmap(edge[0], edge[1], threshold=5, radius=40)
	mymap.scatter(edge[0], edge[1], c='r', s=150, marker=False, alpha=1)

	mymap.draw("temp.html")


# res = agg_location()
# pairwise_sim_agg(res)
# pairwise_dist_sim_agg(res)
# plot_location_node(res)
# plot_sample()