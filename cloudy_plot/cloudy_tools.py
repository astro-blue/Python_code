import numpy as np

def read_in_ovr_file(filename):
	data = np.genfromtxt(filename,names=True)
	
	return data