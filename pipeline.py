import copy
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import phylogeny_construct_functions as pcf
import data_parsing_functions as dpf
import csv
import sys

def main():
	#D = other()
	with open("{}".format(sys.argv[1]), 'r') as file:
		csvreader = csv.reader(file)
		sequence_lengths , percentages , species = dpf.get_sequence_lengths_and_percentages(csvreader)
	D = pd.DataFrame(dpf.get_weighted_matrix(sequence_lengths,percentages))# , index = species , columns = species)
	
	
	G = pcf.additive_phylogeny(D , len(D.index) + 1)
	
	relabel = {}
	for i , animal in enumerate(species):
		relabel[i] = animal
	nx.relabel_nodes(G , relabel , False)

	
	pcf.show(G)
	plt.show()








if __name__ == '__main__':
	main()