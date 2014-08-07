# This file includes some functions to read files

#-------------------------------------
def read_simple_txt_file(index_file):

	# use find xxx > index_file
	# read txt files with list of filename

	f = open(index_file)
	content = f.read().splitlines()
	
	return content
#-------------------------------------	
def read_data_file_into_dict(filename):

	# use to convert simple txt data into dictionary structure
	# might need to be modified
	headers = None
	data = dict()
	for line in open(filename):
		if line[0] == "|":
			if headers == None:
				line = line.replace("|", " ")
				headers = line.split()
				for header in headers:
					data[header] = list()
			continue

		values = line.split()
	
		for index, header in enumerate(headers):
			data[header].append(values[index])
		
		return data
#-----------------------------------------