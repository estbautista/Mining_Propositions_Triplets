import numpy as np
import scipy as sp
import networkx as nx
import utils
import propositions as prop
import decomposition as dec
import matplotlib.pyplot as plt

#################
### READ DATA ###
#################
edgelist_fn = '/Volumes/SSD_Backup/Datasets/Multiplex_Networks/Europe_Airports/edgelist.txt'
nodesIDs_fn = '/Volumes/SSD_Backup/Datasets/Multiplex_Networks/Europe_Airports/nodes_names.txt'
layerIDs_fn = '/Volumes/SSD_Backup/Datasets/Multiplex_Networks/Europe_Airports/layer_names.txt'
edgelist_raw = np.loadtxt(edgelist_fn, dtype=str, delimiter=' ', usecols=[1,2,0])
nodesIDs_raw = np.loadtxt(nodesIDs_fn, dtype=str, delimiter=' ', usecols=[0,1,2,3], skiprows=1)
layerIDs_raw = np.loadtxt(layerIDs_fn, dtype=str, delimiter=' ', usecols=[0,1], skiprows=1)

network_list, nodeIDs_dict, layerIDs_dict, coord_dict = utils.process_airport_data(edgelist_raw, nodesIDs_raw, layerIDs_raw)
A = utils.generate_tensor(network_list)

########################
### GET PROPOSITIONS ###
########################
x_target = 1
y_target = 1
direction = 'C'
prop_dict = prop.get_propositions(A, x_target, y_target, direction)

print('Number of patterns found', len(prop_dict))
prop.print_statements(prop_dict, Role_dict, new_dict, direction, min_size=3)
