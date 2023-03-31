import numpy as np
import itertools
from itertools import chain, combinations

def powerset(my_list):
    return list(chain.from_iterable(combinations(my_list, r) for r in range(1, len(my_list)+1)))

class proposition_obj:
	def __init__(self, x, y, alpha_idx, beta_idx, gamma_idx):
		self.x = x
		self.y = y
		self.alpha_set = alpha_idx
		self.beta_set = beta_idx
		self.gamma_set = gamma_idx

def statement(F, i_idx, j_idx, k_idx, direction):
	region_sum = np.sum( F[np.ix_(i_idx, j_idx, k_idx)] )
	if direction == 'A':
		total_sum = np.sum( F[i_idx, :, :] )
		if total_sum != 0: x = region_sum / total_sum
		else: x = 0

	elif direction == 'B':
		total_sum = np.sum( F[:, j_idx, :] )
		if total_sum != 0: x = region_sum / total_sum
		else: x = 0

	else:
		total_sum = np.sum( F[:, :, k_idx] )
		if total_sum != 0: x = region_sum / total_sum
		else: x = 0

	y = region_sum / (i_idx.size*j_idx.size*k_idx.size)
	return x, y

def predicate(region, A):
	# curr sets
	row_set = list(region[0])
	col_set = list(region[1])

	total = np.sum(A)
	region = np.sum( A[ np.ix_(row_set, col_set) ] )

	x = region/total
	y = region/(len(row_set)*len(col_set))
	return x, y


def evaluate_statement(F, slice_idx, slice_row_set, slice_col_set, direction):
	# We have to test if current set satisfies the statement
	if direction == 'A':
		a_set = slice_idx
		b_set = slice_row_set
		c_set = slice_col_set
		x, y = statement(F, np.array(a_set), np.array(b_set), np.array(c_set), direction)
	elif direction == 'B':
		a_set = slice_row_set
		b_set = slice_idx
		c_set = slice_col_set
		x, y = statement(F, np.array(a_set), np.array(b_set), np.array(c_set), direction)
	else:
		a_set = slice_row_set
		b_set = slice_col_set
		c_set = slice_idx
		x, y = statement(F, np.array(a_set), np.array(b_set), np.array(c_set), direction)
	return x, y, a_set, b_set, c_set

def evaluate_predicate(data_slice, slice_idx, region, direction):
	# We have to test if current set satisfies the statement
	if direction == 'A':
		a_set = slice_idx
		b_set = list(region[0])
		c_set = list(region[1])
		x, y = predicate(region, data_slice)
	elif direction == 'B':
		a_set = list(region[0])
		b_set = slice_idx
		c_set = list(region[1])
		x, y = predicate(region, data_slice)
	else:
		a_set = list(region[0])
		b_set = list(region[1])
		c_set = slice_idx
		x, y = predicate(region, data_slice)
	return x, y, a_set, b_set, c_set


def region_expansion(region, A):
	# curr sets
	row_set = list(region[0])
	col_set = list(region[1])

	# New regions
	row_regions = []
	col_regions = []
	
	# Complement of current regions
	row_set_comp = list(set(range(A.shape[0])).difference(region[0]))
	col_set_comp = list(set(range(A.shape[1])).difference(region[1]))
	
	# Get expansion rows and cols
	edges_per_row = np.sum( A[ np.ix_( row_set_comp, col_set ) ], axis=1)
	edges_per_col = np.sum( A[ np.ix_( row_set, col_set_comp ) ], axis=0)

	# Deciding which direction to expand
	max_row_density = np.max(edges_per_row)/len(col_set)
	max_col_density = np.max(edges_per_col)/len(row_set)

	if max_row_density == 0 and max_col_density == 0: return []
	
	# Expand rows
	max_row_idx = np.where( edges_per_row == np.max(edges_per_row) )[0]
	for x in max_row_idx:
		new_set = row_set + [row_set_comp[x]]
		row_regions.append( (frozenset(new_set), region[1]) )
	# Expand cols
	max_col_idx = np.where( edges_per_col == np.max(edges_per_col) )[0]
	for x in max_col_idx:
		new_set = col_set + [col_set_comp[x]]
		col_regions.append( (region[0], frozenset(new_set)) )	

	# Return based on the densities
	if max_row_density == max_col_density:
		return row_regions + col_regions
	elif max_row_density > max_col_density:
		return row_regions
	else:
		return col_regions


def get_propositions(F, x_target, y_target, direction):

	G_dic = {}
	if direction == 'A': axis_size = F.shape[0]
	elif direction == 'B': axis_size = F.shape[1]
	else: axis_size = F.shape[2]

	for slice_idx in range(axis_size):
		print('-- Processing slice ', slice_idx)
		# The slice keeps the left-most axis as rows and the right-most axis as cols
		if direction == 'A': curr_slice = F[slice_idx, :, :]
		elif direction == 'B': curr_slice = F[:, slice_idx, :]
		else: curr_slice = F[:, :, slice_idx]

		# The edges for the slice form the initial regions
		regions = set()
		for i in range(curr_slice.shape[0]):
			for j in range(curr_slice.shape[1]):
				if curr_slice[i,j] > 0:
					curr_region = (frozenset([i]), frozenset([j]))
					regions.add( curr_region )
		
		# Increase the regions until none of them can be increased
		while len(regions):
			# Expand regions	
			new_regions = set()
			for reg in regions:
				exp = region_expansion(reg, curr_slice)
				print(len(exp))
				for r in exp: 
					new_regions.add(r)
			
			# Evaluate regions
			regions = new_regions.copy()
			new_regions = set()
			rn = 0
			for reg in regions:	
				rn += 1
				slice_num = [slice_idx]
				x, y, a_set, b_set, c_set = evaluate_predicate(curr_slice, slice_num, reg, direction)

				if y >= y_target:
					new_regions.add( reg )
					if x >= x_target:
						p_obj = proposition_obj(x, y, a_set, b_set, c_set)
						if reg not in G_dic.keys(): G_dic[reg] = []
						G_dic[reg].append( p_obj )
			
			# Update regions
			regions = new_regions.copy()
	
	print('Predicate search done..')
	print('Number of predicates retained :', len(G_dic))
	input('--- Press to continue --')
			
	count = 0
	for d_key in G_dic.keys():
		idx_to_join = []
		for obj in G_dic[d_key]:
			if direction == 'A': idx_to_join += obj.alpha_set
			elif direction == 'B': idx_to_join += obj.beta_set
			else: idx_to_join += obj.gamma_set
		
		slice_row_set = list(d_key[0])
		slice_col_set = list(d_key[1])
		x, y, a_set, b_set, c_set = evaluate_statement(F, idx_to_join, slice_row_set, slice_col_set, direction)
		s_obj = proposition_obj(x, y, a_set, b_set, c_set)
		G_dic[d_key].append(s_obj)
		
		count += 1
	return G_dic


def print_statements(prop_dict, nodeIDs, layerIDs, direction, min_size=1):
	file_obj = open('pattern.txt', 'w')
	file_obj = open('pattern.txt', 'a')
	for k in prop_dict.keys():
		for l in prop_dict[k]:
			alpha_to_print = [(nodeIDs[x], x) for x in l.alpha_set]
			beta_to_print = [(nodeIDs[x], x) for x in l.beta_set]
			gamma_to_print = [(layerIDs[x], x) for x in l.gamma_set]
			if direction == 'A':
				if len(alpha_to_print) < min_size: continue
				to_print = ('-----------\n' + str(l.x) + ' % \n' + str(alpha_to_print) +
							'\nis ' + str(l.y) + ' % \n' + str(beta_to_print) +
							'\nand \n' + str(gamma_to_print) + '\n\n')
			elif direction == 'B':
				if len(beta_to_print) < min_size: continue
				to_print = ('-----------\n' + str(l.x) + ' % \n' + str(beta_to_print) +
							'\nis ' + str(l.y) + ' % \n' + str(alpha_to_print) +
							'\nand \n' + str(gamma_to_print) + '\n\n')
			else:	
				if len(gamma_to_print) < min_size: continue
				to_print = ('-----------\n' + str(l.x) + ' % \n' + str(gamma_to_print) +
							'\nis ' + str(l.y) + ' % \n' + str(alpha_to_print) +
							'\nand \n' + str(beta_to_print) + '\n\n')

			file_obj.write(to_print)
