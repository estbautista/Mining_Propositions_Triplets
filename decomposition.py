import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg as spl
import collections
import utils

class region_partition:
	def __init__(self, alpha, beta, gamma, A_size, B_size, C_size):
		self.Left_tree = None
		self.Right_tree = None
		self.Psi_pos = None
		self.Psi_neg = None
		self.alpha_set = alpha
		self.beta_set = beta
		self.gamma_set = gamma
		self.A_size = A_size
		self.B_size = B_size
		self.C_size = C_size
		self.Region_size = len(alpha)*len(beta)*len(gamma)
	
	def create_subtree(self, A_tree, B_tree, C_tree, update):
		if update == 'A':
			self.Left_tree = region_partition(A_tree.Left_tree.Set, B_tree.Set, C_tree.Set, self.A_size, self.B_size, self.C_size)
			self.Right_tree = region_partition(A_tree.Right_tree.Set, B_tree.Set, C_tree.Set, self.A_size, self.B_size, self.C_size)
	
			if B_tree.Left_tree != None: 
				self.Left_tree.create_subtree(A_tree.Left_tree, B_tree, C_tree, 'B')
				self.Right_tree.create_subtree(A_tree.Right_tree, B_tree, C_tree, 'B')
			elif C_tree.Left_tree != None:
				self.Left_tree.create_subtree(A_tree.Left_tree, B_tree, C_tree, 'C')
				self.Right_tree.create_subtree(A_tree.Right_tree, B_tree, C_tree, 'C')
			elif A_tree.Left_tree.Left_tree != None: 
				self.Left_tree.create_subtree(A_tree.Left_tree, B_tree, 'A')
				self.Right_tree.create_subtree(A_tree.Right_tree, B_tree, 'A')

		elif update == 'B':
			self.Left_tree = region_partition(A_tree.Set, B_tree.Left_tree.Set, C_tree.Set, self.A_size, self.B_size, self.C_size)
			self.Right_tree = region_partition(A_tree.Set, B_tree.Right_tree.Set, C_tree.Set, self.A_size, self.B_size, self.C_size)

			if C_tree.Left_tree != None:
				self.Left_tree.create_subtree(A_tree, B_tree.Left_tree, C_tree, 'C')
				self.Right_tree.create_subtree(A_tree, B_tree.Right_tree, C_tree, 'C')
			elif A_tree.Left_tree != None:
				self.Left_tree.create_subtree(A_tree, B_tree.Left_tree, C_tree, 'A')
				self.Right_tree.create_subtree(A_tree, B_tree.Right_tree, C_tree, 'A')
			elif B_tree.Left_tree.Left_tree != None:
				next_update = 'B'
				self.Left_tree.create_subtree(A_tree, B_tree.Left_tree, C_tree, 'B')
				self.Right_tree.create_subtree(A_tree, B_tree.Right_tree, C_tree, 'B')

		elif update == 'C':
			self.Left_tree = region_partition(A_tree.Set, B_tree.Set, C_tree.Left_tree.Set, self.A_size, self.B_size, self.C_size)
			self.Right_tree = region_partition(A_tree.Set, B_tree.Set, C_tree.Right_tree.Set, self.A_size, self.B_size, self.C_size)

			if A_tree.Left_tree != None:
				self.Left_tree.create_subtree(A_tree, B_tree, C_tree.Left_tree, 'A')
				self.Right_tree.create_subtree(A_tree, B_tree, C_tree.Right_tree, 'A')
			elif B_tree.Left_tree != None:
				self.Left_tree.create_subtree(A_tree, B_tree, C_tree.Left_tree, 'B')
				self.Right_tree.create_subtree(A_tree, B_tree, C_tree.Right_tree, 'B')
			elif C_tree.Left_tree.Left_tree != None:
				self.Left_tree.create_subtree(A_tree, B_tree, C_tree.Left_tree, 'C')
				self.Right_tree.create_subtree(A_tree, B_tree, C_tree.Right_tree, 'C')	

		self.Psi_pos = [ x * self.B_size + y + z*(self.A_size*self.B_size) for z in self.Left_tree.gamma_set for x in self.Left_tree.alpha_set for y in self.Left_tree.beta_set ]
		self.Psi_neg = [ x * self.B_size + y + z*(self.A_size*self.B_size) for z in self.Right_tree.gamma_set for x in self.Right_tree.alpha_set for y in self.Right_tree.beta_set ]	

class set_partition:
	def __init__(self, elems, root_size):
		self.Left_tree = None
		self.Right_tree = None
		self.Set = elems
		self.Root_size = root_size
	
	def create_subtree(self, base_field, max_levels):
		if max_levels != None:
			min_size = max(1, self.Root_size//(2**max_levels))
		else:
			min_size = 1
		if len(self.Set) == min_size:
			return
		else:
			set1, set2 = self.partition_set( base_field )	

			self.Left_tree = set_partition( set1, self.Root_size )
			self.Left_tree.create_subtree( base_field, max_levels )

			self.Right_tree = set_partition( set2, self.Root_size )
			self.Right_tree.create_subtree( base_field, max_levels )

	def partition_set(self, base_field):
		matrix_region = base_field[self.Set, :]	
		u, _, _ = spl.svds(matrix_region, k=1, which='LM')
		
		sort_indices = np.argsort(-u.flatten())
		
		set1 = [self.Set[i] for i in sort_indices[: sort_indices.size//2]]
		set2 = [self.Set[i] for i in sort_indices[sort_indices.size//2 :]]
		return set1, set2

def space_partitioning(network_list, max_levels=None):

	A_mode = utils.mode_reshape(network_list, 'A')
	B_mode = utils.mode_reshape(network_list, 'B')
	C_mode = utils.mode_reshape(network_list, 'C')
		
	# Extract initial sets
	A_set = list(range(A_mode.shape[0]))
	B_set = list(range(B_mode.shape[0]))
	C_set = list(range(C_mode.shape[0]))

	# Create partition tree of set A
	A_tree = set_partition(A_set, len(A_set)) 
	A_tree.create_subtree(A_mode, max_levels)	

	# Create partition tree of set B
	B_tree = set_partition(B_set, len(B_set))
	B_tree.create_subtree(B_mode, max_levels)

	# Create partition tree of set C
	C_tree = set_partition(C_set, len(C_set))
	C_tree.create_subtree(C_mode, max_levels)
	
	return A_tree, B_tree, C_tree

def region_tree_generation(A_tree, B_tree, C_tree):
	
	# Initial set sizes
	A_size = len(A_tree.Set)
	B_size = len(B_tree.Set)
	C_size = len(C_tree.Set)
	
	# Create the root of the tree
	region_tree = region_partition(A_tree.Set, B_tree.Set, C_tree.Set, A_size, B_size, C_size)

	# Create the subtree of regions
	region_tree.create_subtree(A_tree, B_tree, C_tree, update='A')

	return region_tree

def get_dictionary(region_tree):
	row_index = []
	col_index = []
	values = []
	q = collections.deque([region_tree])

	counter = 0
	while len(q):
		curr_node = q[0]
		
		# Add positive indices
		col_index += curr_node.Psi_pos
		values += [1/np.sqrt(curr_node.Region_size)]*(curr_node.Region_size//2)

		# Add negative indices
		col_index += curr_node.Psi_neg
		values += [-1/np.sqrt(curr_node.Region_size)]*(curr_node.Region_size//2) 

		# Add the index of the current basis element
		row_index += [counter]*curr_node.Region_size 

		# Update the queue
		counter += 1
		q.popleft()
		if (curr_node.Left_tree.Psi_pos != None):
			q.append( curr_node.Left_tree )
			q.append( curr_node.Right_tree )
	
	basis_matrix = sps.coo_matrix( (values, (row_index, col_index)), shape=(region_tree.Region_size, region_tree.Region_size) ).tocsr()
	return basis_matrix

def graph_decomposition(graph, basis):	
	mean_val = graph.sum(axis=1)/np.sqrt(graph.shape[1])
	graph_dec = graph.dot(basis.T)
	struct_coef = sps.hstack([mean_val, graph_dec])
	return struct_coef

def update_tree_coefficients(decomp, region_tree):		

	# Store the first coef 
	decomp = np.array(decomp.todense()).flatten()
	region_tree.Coefficient = decomp[0]/np.sqrt(region_tree.Region_size)
	region_tree.Density = region_tree.Coefficient

	# Create queue and counter
	q = collections.deque([region_tree])
	counter = 1
	while len(q):
		curr_node = q[0]

		curr_node.Left_tree.Coefficient = decomp[counter]/np.sqrt(curr_node.Region_size)
		curr_node.Left_tree.Density = curr_node.Density + curr_node.Left_tree.Coefficient

		curr_node.Right_tree.Coefficient = -decomp[counter]/np.sqrt(curr_node.Region_size)
		curr_node.Right_tree.Density = curr_node.Density + curr_node.Right_tree.Coefficient

		q.popleft()
		counter += 1
		if (curr_node.Left_tree.Left_tree != None):
			q.append( curr_node.Left_tree )
			q.append( curr_node.Right_tree )

	return region_tree
