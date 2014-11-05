import read_files as read_files
import pyfits as pf
import numpy as np
reload(read_files)
index_filename = read_files.read_simple_txt_file('index_file.txt')

for i_filename in index_filename:
	output = read_files.read_csv_file_into_dict(i_filename)
	list_fits = ['ra','dec','glat','glon','e_bv','fuv_mag','fuv_magerr','nuv_mag','nuv_magerr']
	col_def = []
	for i_list_fits in list_fits:
		tmp_col = pf.Column(name=i_list_fits,format='D',array=map(float,output[i_list_fits]))
		col_def.append(tmp_col)
	cols=pf.ColDefs(col_def)
	my_table = pf.new_table(cols)
	my_table.writeto(i_filename.replace('csv','fits'),clobber=True)