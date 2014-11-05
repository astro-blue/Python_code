import numpy as np

def read_in_ovr_file(filename,n_b,headers):
	if headers==True:
		data = np.genfromtxt(filename,names=True,usecols=(n_b))
	else:
		data = np.genfromtxt(filename,usecols=(n_b))
	
	return data