import cloudy_tools as c_tools
import make_plot_set_up as my_set_up
import matplotlib.pyplot as plt
def plot_HW():
	data = c_tools.read_in_ovr_file('hii_paris.ovr')
	
	my_set_up.make_plot_set_up({'setting':0,'fontsize':15,'lw':2})
	
	
	plt.plot(data['depth']/10.**19,np.log10(data['Te']),lw=3,color='black')
	plt.ylim(3.01,4.3)
	plt.xlabel('depth [$\\rm 10^{19} cm$]',fontsize=18)
	plt.ylabel('log $(T_{e}/K)$',fontsize=18)
	plt.savefig('TingWen_cloudy_HW_2.pdf',clobber=True)
plot_HW()
