import scipy.sparse as sps
import scipy as sp
import numpy as np
import math
import collections
import folium
import webbrowser
import os

def mode_reshape(network_list, mode):
	
	if mode == 'A':
		mat = sps.hstack( network_list ).tocsr().asfptype()

	elif mode == 'B':
		mat = sps.vstack( network_list )
		mat = mat.T.tocsr().asfptype()

	elif mode == 'C':
		flatten_networks = [x.reshape(1, x.shape[0]*x.shape[1]) for x in network_list ]
		mat = sps.vstack( flatten_networks ).tocsr().asfptype()

	return mat

def generate_tensor(network_list):
	data_cube = np.zeros([network_list[0].shape[0], network_list[0].shape[1], len(network_list)])
	for i in range(len(network_list)):
		data_cube[:,:,i] = network_list[i].todense()
	return data_cube

def flatten_network(network_list):
	flatten_net_list = [x.reshape(1, x.shape[0]*x.shape[1]) for x in network_list ]
	mat = sps.hstack( flatten_net_list )
	return mat

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

def process_hospital_data(raw_dataset):

	# Map agents names to numerical matrix indices
	Node_dict = {}
	Time_dict = {}
	for row	in range(raw_dataset.shape[0]):
		# Interaction information
		time = raw_dataset[row, 0]
		agent_1 = raw_dataset[row, 1]
		agent_2 = raw_dataset[row, 2]
		role_agent_1 = raw_dataset[row, 3]
		role_agent_2 = raw_dataset[row, 4]
		
		# Store data
		if agent_1 not in Node_dict.keys(): 
			Node_dict[agent_1] = {'ID':len(Node_dict), 'Role':role_agent_1}
		if agent_2 not in Node_dict.keys():
			Node_dict[agent_2] = {'ID':len(Node_dict), 'Role':role_agent_2}
		if time not in Time_dict.keys():
			Time_dict[time] = len(Time_dict)
	
	# Extract the role associated to numerical indices of agents
	Role_dict = {}
	for agent in Node_dict.keys():
		agent_index = Node_dict[agent]['ID']
		agent_role = Node_dict[agent]['Role']
		Role_dict[agent_index] = agent_role

	# Extract the time associated to numerical indices of layers
	Layer_dict = {}
	for time in Time_dict.keys():
		Layer_dict[ Time_dict[time] ] = time

	# Extract dataset info
	set_size = len(Node_dict)
	layer_size = len(Time_dict)

	edge_lists_dict = {}
	for row in range(raw_dataset.shape[0]):
		elem_A = raw_dataset[row, 1]
		elem_B = raw_dataset[row, 2]
		elem_C = raw_dataset[row, 0]		

		curr_layer = Time_dict[elem_C]
		if curr_layer not in edge_lists_dict.keys(): edge_lists_dict[curr_layer] = {'source':[], 'destination':[]}
		edge_lists_dict[curr_layer]['source'].append(Node_dict[elem_A]['ID'])
		edge_lists_dict[curr_layer]['destination'].append(Node_dict[elem_B]['ID'])	
		edge_lists_dict[curr_layer]['source'].append(Node_dict[elem_B]['ID'])
		edge_lists_dict[curr_layer]['destination'].append(Node_dict[elem_A]['ID'])
		
	network_list = []
	for i in range(layer_size):
		if i in edge_lists_dict.keys():
			source = edge_lists_dict[i]['source']
			destin = edge_lists_dict[i]['destination']
			adj = sps.coo_matrix( ([1]*len(source), (source, destin)), shape=(set_size, set_size) ).tocsr()
		else:
			adj = sps.csr_matrix((set_size, set_size))
		network_list.append( adj )
	
	return network_list, Role_dict, Layer_dict

def process_airport_data(data, nodesIDs, layerIDs):
	# Map each element of the sets A B C to a numerical index to represent it in a sequence of matrices
	indices_A = {} # Set of source is going to be equal to destination by construction of the dataset
	indices_C = {} # Set of layers
	for row in range(data.shape[0]):
		elem_A = data[row, 0]
		elem_B = data[row, 1]
		elem_C = data[row, 2]
		if elem_A not in indices_A.keys(): indices_A[elem_A] = len(indices_A)
		if elem_B not in indices_A.keys(): indices_A[elem_B] = len(indices_A)
		if elem_C not in indices_C.keys(): indices_C[elem_C] = len(indices_C)
	indices_B = indices_A

	len_setA = len(indices_A)
	len_setB = len(indices_B)
	len_setC = len(indices_C)

	nodeIDs_dict = {}
	coord_dict = {}
	for row in range(nodesIDs.shape[0]):
		init_ID = nodesIDs[row, 0]
		node_name =	nodesIDs[row, 1]
		longitude = float(nodesIDs[row, 2])
		latitude = float(nodesIDs[row, 3])
		if init_ID not in indices_A.keys(): continue
		else: 
			nodeIDs_dict[ indices_A[init_ID] ] = node_name
			coord_dict[ indices_A[init_ID] ] = [latitude, longitude]
	for i in range(len(nodeIDs_dict), len_setA):
		nodeIDs_dict[i] = 'UNDEF'
		coord_dict[i] = [0.0, 0.0]
	
	layerIDs_dict = {}
	for row in range(layerIDs.shape[0]):
		init_ID = layerIDs[row, 0]
		layer_name = layerIDs[row, 1]
		if init_ID not in indices_C.keys(): continue
		else: layerIDs_dict[ indices_C[init_ID] ] = layer_name
	for i in range(len(layerIDs_dict), len_setC):
		layerIDs_dict[i] = 'UNDEF'
		
	edge_lists_dict = {}
	for row in range(data.shape[0]):
		elem_A = data[row, 0]
		elem_B = data[row, 1]
		elem_C = data[row, 2]		

		curr_layer = indices_C[elem_C]
		if curr_layer not in edge_lists_dict.keys(): edge_lists_dict[curr_layer] = {'source':[], 'destination':[]}
		edge_lists_dict[curr_layer]['source'].append(indices_A[elem_A])
		edge_lists_dict[curr_layer]['destination'].append(indices_B[elem_B])	
		edge_lists_dict[curr_layer]['source'].append(indices_B[elem_B])
		edge_lists_dict[curr_layer]['destination'].append(indices_A[elem_A])
		
	network_list = []
	for i in range(len_setC):
		if i in edge_lists_dict.keys():
			source = edge_lists_dict[i]['source']
			destin = edge_lists_dict[i]['destination']
			adj = sps.coo_matrix( ([1]*len(source), (source, destin)), shape=(len_setA, len_setB) ).tocsr()
		else:
			adj = sps.csr_matrix((len_setA, len_setB))
		network_list.append( adj )
	
	return network_list, nodeIDs_dict, layerIDs_dict, coord_dict

def get_region_info(r_num, region_tree, n_IDs, l_IDs, coords):
	counter = 0
	q = collections.deque([region_tree])
	while counter < r_num:
		curr_node = q[0]
		q.popleft()
		counter += 1
		if (curr_node.Left_tree != None):
			q.append( curr_node.Left_tree )
			q.append( curr_node.Right_tree )

	pos_alpha = curr_node.Left_tree.alpha_set
	pos_beta = curr_node.Left_tree.beta_set
	pos_gamma = curr_node.Left_tree.gamma_set

	neg_alpha = curr_node.Right_tree.alpha_set
	neg_beta = curr_node.Right_tree.beta_set
	neg_gamma = curr_node.Right_tree.gamma_set

	print('pos_alpha = ', [n_IDs[x] for x in pos_alpha])
	print('\n')
	print('pos_beta = ', [n_IDs[x] for x in pos_beta])
	print('\n')
	print('pos_gamma = ', [l_IDs[x] for x in pos_gamma])

	print('\n\n')
	print('neg_alpha = ', [n_IDs[x] for x in neg_alpha])
	print('\n')
	print('neg_beta = ', [n_IDs[x] for x in neg_beta])
	print('\n')
	print('neg_gamma = ', [l_IDs[x] for x in neg_gamma])


	coords_alpha = np.array([coords[x] for x in pos_alpha])
	coords_beta = np.array([coords[x] for x in pos_beta])
	coords_gamma = [l_IDs[x] for x in pos_gamma]

	my_map = folium.Map(location=[coords_alpha[0,0], coords_alpha[0,1]], zoom_start=3)
	for i in range(coords_alpha.shape[0]):
		folium.Marker([coords_alpha[i,0], coords_alpha[i,1]], popup=coords_gamma, icon=folium.Icon(color='red')).add_to(my_map)
	for i in range(coords_beta.shape[0]):
		folium.Marker([coords_beta[i,0], coords_beta[i,1]], popup=coords_gamma, icon=folium.Icon(color='green')).add_to(my_map)

	my_map.save("map_1.html")
	webbrowser.open_new("file://" + os.path.realpath("map_1.html"))

	coords_alpha = np.array([coords[x] for x in neg_alpha])
	coords_beta = np.array([coords[x] for x in neg_beta])
	coords_gamma = [l_IDs[x] for x in neg_gamma]

	my_map = folium.Map(location=[coords_alpha[0,0], coords_alpha[0,1]], zoom_start=3)
	for i in range(coords_alpha.shape[0]):
		folium.Marker([coords_alpha[i,0], coords_alpha[i,1]], popup=coords_gamma, icon=folium.Icon(color='blue')).add_to(my_map)
	for i in range(coords_beta.shape[0]):
		folium.Marker([coords_beta[i,0], coords_beta[i,1]], popup=coords_gamma, icon=folium.Icon(color='black')).add_to(my_map)

	my_map.save("map_2.html")
	webbrowser.open_new("file://" + os.path.realpath("map_2.html"))
