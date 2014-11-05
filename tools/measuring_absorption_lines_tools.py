import pyfits as pf
import numpy as np
from kapteyn import kmpfit

def check_input_parameter


		cut = np.where((fitting_x<DIB_info['exclude'][0]) | (fitting_x>DIB_info['exclude'][1]))
		fitobj = kmpfit.Fitter(residuals=residuals_single_g,data=(fitting_x[cut],fitting_y[cut],fitting_yerr[cut]))

		if fixed_width == -1:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-1,para[1]+1)},
				{'limits':(para[2]-width_range*para[2],para[2]+width_range*para[2])},\
				{'limits':(para[3]-0.1,para[3]+0.1)})
			
			
		if fixed_width >= 0:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-0.5,para[1]+0.5)},
				{'fixed':True},\
				{'limits':(para[3]-0.1,para[3]+0.1)})
			#print '?'
		


		fitobj.fit(params0=para)	
		fit_para = fitobj.params
		perr = fitobj.stderr
		DIB_output = {}
		area,area_error = give_me_area(fit_para[0],fit_para[2],perr[0],perr[2])
		DIB_output[0] = {'fit_para':fit_para,'fit_err':perr,'area':[area,area_error]}





























def DIB_fitting_info(name):
	exclude = [0.0]
	if name=='4430':
		fitting_type = 1 # 1=single 2=double 3=three
		
		amplitude = 0.2
		center = 4428.298
		width = 7.74
		zerop = 1.0
		para = [amplitude,center,width,zerop]
		range_for_line = {'L':[para[1]-40,para[1]-30],'R':[para[1]+30,para[1]+40]}	
		dib_name = ['4430']
	if name=='5780':
		fitting_type = 3
		range_for_line = {'L':[5750,5765],'R':[5804,5810]}
		para = [0.1,5779.96,5.69,0.1,5780.81,1.611,0.1,5796.91,1.65,1.0]
		dib_name = ['5778','5780','5797']
	if name=='5781':
		fitting_type = 2	
		para = [0.1,5780,1.5,0.1,5797,1.5,1.0]
		range_for_line = {'L':[para[1]-10,para[1]-5],'R':[para[4]+5,para[4]+10]}
		dib_name = ['5780','5797']
	if name=='6283':
		fitting_type = 2
		para = [0.1,6269.65,1.276,0.1,6285.2,3.116,1.0]
		range_for_line = {'L':[para[1]-15,para[1]-10],'R':[para[4]+10,para[4]+15]}
		dib_name = ['6270','6283']

	if name=='6284':
		fitting_type = 5
		para = [0.1,6285.15,2.924,1.0]
		range_for_line = {'L':[para[1]-30,para[1]-25],'R':[para[1]+10,para[1]+15]}
		dib_name = ['6283']
		exclude = [6265,6275]



	if name=='6614':
		fitting_type = 1
		para = [0.1,6613.929,1.959,1.0]
		range_for_line = {'L':[para[1]-15,para[1]-10],'R':[para[1]+10,para[1]+15]}
		dib_name = ['6614']
	if name=='4502':
		fitting_type = 1
		para= [0.1,4501.818,2.098,1.0]	
		range_for_line = {'L':[para[1]-10,para[1]-5],'R':[para[1]+5,para[1]+10]}
		dib_name = ['4502']
	if name=='4728':
		fitting_type = 1
		para= [0.1,4726.7889,1.88,1.0]	
		range_for_line = {'L':[para[1]-15,para[1]-8],'R':[para[1]+8,para[1]+15]}
		dib_name = ['4728']
	if name=='4770':
		fitting_type = 2
		para= [0.1,4762,1.0,0.1,4762,5.0,1.0]	
		range_for_line = {'L':[4740,4750],'R':[4785,4795]}
	if name=='4771':
		fitting_type = 1
		para= [0.1,4762.722,1.85,1.0]	
		range_for_line = {'L':[para[1]-10,para[1]-5],'R':[para[1]+10,para[1]+15]}
		dib_name = ['4762']
	if name=='4772':
		fitting_type = 7
		para= [0.1,4762.722,1.85,0.1,4765.0,2.0,1.0]	
		range_for_line = {'L':[para[1]-10,para[1]-5],'R':[para[1]+10,para[1]+15]}
		dib_name = ['4762']


	if name=='5850':
		fitting_type = 2
		para= [0.1,5844.01,2.20,0.1,5850.10,1.7,1.0]	
		range_for_line = {'L':[para[1]-10,para[1]-5],'R':[para[4]+5,para[4]+10]}
		dib_name = ['5845','5850']
	if name=='5852':
		fitting_type = 7
		para= [0.1,5845,4.0,0.1,5850,4.0,1.0]	
		range_for_line = {'L':[para[1]-10,para[1]-5],'R':[para[4]+5,para[4]+10]}
		dib_name = ['5850']
	

	if name=='5851':
		fitting_type = 1
		para= [0.1,5848,5.0,1.0]	
		range_for_line = {'L':[para[1]-20,para[1]-15],'R':[para[1]+15,para[1]+20]}
		dib_name = ['5850']
	if name=='5705':
		fitting_type = 1
		para= [0.1,5705.40,2.15,1.0]	
		range_for_line = {'L':[para[1]-15,para[1]-10],'R':[para[1]+10,para[1]+15]}
		dib_name = ['5705']
	if name=='4885':
		fitting_type = 5
		para =[0.1,4883.81,11.4,1.0]
		range_for_line = {'L':[para[1]-70,para[1]-50],'R':[para[1]+50,para[1]+70]}
		exclude = [4855,4865]
		dib_name = ['4885']
	if name=='4960':
		fitting_type = 1
		para =[0.1,4960,5,1.0]
		range_for_line = {'L':[para[1]-30,para[1]-20],'R':[para[1]+20,para[1]+30]}
		exclude = [4855,4865]
		dib_name = ['4960']
	if name=='5550':
		fitting_type=4
		para = [0.1,5488,3.0,0.1,5508,1.0,0.1,5526,1.0,0.1,5540,5,1.0]	
		range_for_line = {'L':[5470,5475],'R':[5560,5570]}
		#dib_name = ['54']
	if name=='5551':
		fitting_type=3
		para = [0.1,5487.95,2.31,0.1,5507.858,1.86,0.1,5535.6,10.15,1.0]	
		range_for_line = {'L':[5465,5475],'R':[5560,5570]}
		dib_name = ['5487','5508','5540']
	if name=='6010':
		fitting_type=1
		para = [0.1,6010.56,2.367,1.0]	
		range_for_line = {'L':[para[1]-16,para[1]-8],'R':[para[1]+8,para[1]+16]}
		dib_name = ['6010']
	if name=='6204':
		fitting_type=2
		para = [0.1,6203.568,1.0,0.1,6196,1.0,1.0]	
		range_for_line = {'L':[6186,6190],'R':[6220,6225]}
		dib_name = ['6204','6196']
	if name=='6203':
		fitting_type=5
		para = [0.1,6203.568,2.12,1.0]	
		#range_for_line = {'L':[6186,6190],'R':[6220,6225]}
		range_for_line = {'L':[6155,6160],'R':[6210,6225]}

		dib_name = ['6204']
		exclude = [6193,6198]

	if name=='6380':
		fitting_type=1
		para = [0.1,6378.8,1.84,1.0]	
		range_for_line = {'L':[para[1]-15,para[1]-10],'R':[para[1]+10,para[1]+15]}
		dib_name = ['6379']

	if name=='5448':
		fitting_type=1
		para = [0.1,5449.365,6.02,1.0]	
		range_for_line = {'L':[para[1]-20,para[1]-15],'R':[para[1]+15,para[1]+20]}
		dib_name = ['5448']
	if name=='NaD':
		fitting_type = 2
		para = [0.1,5889.5,3.0,0.1,5896,2.0,1.0]
		range_for_line = {'L':[para[1]-20,para[1]-15],'R':[para[1]+15,para[1]+20]}
		dib_name = ['NaD1','NaD2']
	return {'fitting_type':fitting_type,'para':para,'range_for_line':range_for_line,'exclude':exclude,'dib_name':dib_name}

def give_me_area(amplitude,width,amplitude_err,width_err):
	area = np.sqrt(2*3.14159)*amplitude*np.abs(width)

	
	area_error = area*np.sqrt((amplitude_err/amplitude)**2.0+(width_err/width)**2.0)

	if amplitude_err==0:
		area_error=-99.0

	
	if (amplitude_err==0) & (width_err==0):
		area_error=-99.0
	if (amplitude==0) | (width==0):
		area_error=-99.0
	
	return area, area_error
def go_fitting(wave,spectrum,DIB_info,plot_or_not,fixed_width):
	if fixed_width==-1:
		print 'width varies'
	if fixed_width>=0:
		print 'fixed width'	
	width_range= 1
	# fit a line
	cut = (wave>4300) & (wave<6700)
	sigma,mean = spec_tools.sigma_clipping(spectrum[cut],3)
	#print sigma
	spec_err = np.arange(0,len(wave))*0.0+1
	
	cut_for_line = np.where(((wave>DIB_info['range_for_line']['L'][0]) & (wave<DIB_info['range_for_line']['L'][1])) |\
		((wave>DIB_info['range_for_line']['R'][0]) & (wave<DIB_info['range_for_line']['R'][1])))
	#print wave
	para = [.0,1.0]
	fitobj = kmpfit.Fitter(residuals=residuals_line,data=(wave[cut_for_line[0]],spectrum[cut_for_line[0]],spec_err[cut_for_line[0]]))
	fitobj.fit(params0=para)
	line_para = fitobj.params
	#print line_para
	#-----------------------------------
	#fit gaussian
	cut_for_fitting = np.where((wave>DIB_info['range_for_line']['L'][0]) & (wave<DIB_info['range_for_line']['R'][1]))	
	fitting_x = wave[cut_for_fitting]
	#print fitting_x
	spec_fit = spectrum[cut_for_fitting]
	
	fitting_y = spec_fit/a_line(line_para,fitting_x)
	
	fitting_yerr = spec_err[cut_for_fitting]
	display_y = spectrum
	display_y[cut_for_fitting]=fitting_y
	
	para = DIB_info['para']
	if DIB_info['fitting_type']==1:
		fitobj = kmpfit.Fitter(residuals=residuals_single_g,data=(fitting_x,fitting_y,fitting_yerr))
		
		if fixed_width == -1:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-1,para[1]+1)},
				{'limits':(para[2]-width_range*para[2],para[2]+width_range*para[2])},\
				{'limits':(para[3]-0.1,para[3]+0.1)})

		if fixed_width >= 0:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-0.5,para[1]+0.5)},
				{'fixed':True},\
				{'limits':(para[3]-0.1,para[3]+0.1)})
						

		fitobj.fit(params0=para)	
		fit_para = fitobj.params
		perr = fitobj.stderr
		DIB_output = {}
		area,area_error = give_me_area(fit_para[0],fit_para[2],perr[0],perr[2])
		DIB_output[0] = {'fit_para':fit_para,'fit_err':perr,'area':[area,area_error]}
		
		
		
		
	if DIB_info['fitting_type']==2:
		fitobj = kmpfit.Fitter(residuals=residuals_double_g,data=(fitting_x,fitting_y,fitting_yerr))

		if fixed_width == -1:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-2,para[1]+2)},
				{'limits':(para[2]-width_range*para[2],para[2]+width_range*para[2])},\
				
				{'limits':(-100.0,1.0)},\
				{'limits':(para[4]-2,para[4]+2)},
				{'limits':(para[5]-width_range*para[5],para[5]+width_range*para[5])},\
				{'limits':(para[6]-0.1,para[6]+0.1)})
		if fixed_width >= 0:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-1.0,para[1]+1.0)},
				{'fixed':True},\
				
				{'limits':(-100.0,1.0)},\
				{'limits':(para[4]-1.0,para[4]+1.0)},
				{'fixed':True},\
				{'limits':(para[6]-0.1,para[6]+0.1)})
			
			



		fitobj.fit(params0=para)	
		fit_para = fitobj.params
		perr = fitobj.stderr
		DIB_output = {}
		for i in range(0,2):
			area,area_error = give_me_area(fit_para[0+i*3],fit_para[2+i*3],perr[0+i*3],perr[2+i*3])
			DIB_output[i]={'fit_para':fit_para[0+i*3:3+i*3],'fit_err':perr[0+i*3:3+3*i],'area':[area,area_error]}		
			
		
		
				
	if DIB_info['fitting_type']==3:
		fitobj = kmpfit.Fitter(residuals=residuals_three_g,data=(fitting_x,fitting_y,fitting_yerr))
		if fixed_width == -1:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-1,para[1]+1)},
				{'limits':(para[2]-width_range*para[2],para[2]+width_range*para[2])},\
				
				{'limits':(-100.0,1.0)},\
				{'limits':(para[4]-1,para[4]+1)},
				{'limits':(para[5]-width_range*para[5],para[5]+width_range*para[5])},\
				
				{'limits':(-100.0,1.0)},\
				{'limits':(para[7]-1,para[7]+1)},
				{'limits':(para[8]-width_range*para[8],para[8]+width_range*para[8])},\
				{'limits':(para[9]-0.1,para[9]+0.1)})
		if fixed_width == 0:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-0.5,para[1]+0.5)},
				{'fixed':True},\
				
				{'limits':(-100.0,1.0)},\
				{'limits':(para[4]-0.5,para[4]+0.5)},
				{'fixed':True},\
				
				{'limits':(-100.0,1.0)},\
				{'limits':(para[7]-0.5,para[7]+0.5)},
				{'fixed':True},\
				{'limits':(para[9]-0.1,para[9]+0.1)})
		fitobj.fit(params0=para)	
		fit_para = fitobj.params
		perr = fitobj.stderr	
		DIB_output = {}
		for i in range(0,3):
			area,area_error = give_me_area(fit_para[0+i*3],fit_para[2+i*3],perr[0+i*3],perr[2+i*3])
			DIB_output[i]={'fit_para':fit_para[0+i*3:3+i*3],'fit_err':perr[0+i*3:3+3*i],'area':[area,area_error],'zero':fit_para[-1]}		

	if DIB_info['fitting_type']==4:
		fitobj = kmpfit.Fitter(residuals=residuals_four_g,data=(fitting_x,fitting_y,fitting_yerr))
		fitobj.fit(params0=para)	
		fit_para = fitobj.params
		perr = fitobj.stderr
		DIB_output = {}
		N_g = DIB_info['fitting_type']
		for i in range(0,N_g):
			area,area_error = give_me_area(fit_para[0+i*3],fit_para[2+i*3],perr[0+i*3],perr[2+i*3])
			DIB_output[i]={'fit_para':fit_para[0+i*3:3+i*3],'fit_err':perr[0+i*3:3+3*i],'area':[area,area_error]}		

			

				
	if DIB_info['fitting_type']==5:
		cut = np.where((fitting_x<DIB_info['exclude'][0]) | (fitting_x>DIB_info['exclude'][1]))
		fitobj = kmpfit.Fitter(residuals=residuals_single_g,data=(fitting_x[cut],fitting_y[cut],fitting_yerr[cut]))

		if fixed_width == -1:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-1,para[1]+1)},
				{'limits':(para[2]-width_range*para[2],para[2]+width_range*para[2])},\
				{'limits':(para[3]-0.1,para[3]+0.1)})
			
			
		if fixed_width >= 0:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-0.5,para[1]+0.5)},
				{'fixed':True},\
				{'limits':(para[3]-0.1,para[3]+0.1)})
			#print '?'
		


		fitobj.fit(params0=para)	
		fit_para = fitobj.params
		perr = fitobj.stderr
		DIB_output = {}
		area,area_error = give_me_area(fit_para[0],fit_para[2],perr[0],perr[2])
		DIB_output[0] = {'fit_para':fit_para,'fit_err':perr,'area':[area,area_error]}


	if DIB_info['fitting_type']==7:
		#cut = np.where((fitting_x<DIB_info['exclude'][0]) | (fitting_x>DIB_info['exclude'][1]))
		fitobj = kmpfit.Fitter(residuals=residuals_double_g,data=(fitting_x,fitting_y,fitting_yerr))
		if fixed_width == -1:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-1,para[1]+1)},
				{'limits':(para[2]-width_range*para[2],para[2]+width_range*para[2])},\
				
				{'limits':(-100.0,1.0)},\
				{'limits':(para[4]-1,para[4]+1)},
				{'limits':(para[5]-width_range*para[5],para[5]+width_range*para[5])},\
				{'limits':(para[6]-0.1,para[6]+0.1)})
			
		if fixed_width >= 0:
			fitobj.parinfo = ({'limits':(-100.0,1.0)},\
				{'limits':(para[1]-1,para[1]+1)},
				{'fixed':True},\
				
				{'limits':(-100.0,1.0)},\
				{'limits':(para[4]-1,para[4]+1)},
				{'fixed':True},\
				{'limits':(para[6]-0.1,para[6]+0.1)})
			




		fitobj.fit(params0=para)	
		fit_para = fitobj.params
		perr = fitobj.stderr
		DIB_output = {}
		i=0
		tmp_area_1,tmp_area_error_1 = give_me_area(fit_para[0+i*3],fit_para[2+i*3],perr[0+i*3],perr[2+i*3])
		i=1
		tmp_area_2,tmp_area_error_2 = give_me_area(fit_para[0+i*3],fit_para[2+i*3],perr[0+i*3],perr[2+i*3])
		area = tmp_area_1+tmp_area_2
		area_error = np.sqrt(tmp_area_error_1**2.0+tmp_area_error_2**2.0)
			
		DIB_output[0]={'fit_para':fit_para[0+i*3:3+i*3],'fit_err':perr[0+i*3:3+3*i],'area':[area,area_error]}		
			
		


	if plot_or_not==True:
		plt.plot(wave,display_y,color='black',lw=3,drawstyle='steps-mid')	
	
		if (DIB_info['fitting_type']==1) | (DIB_info['fitting_type']==5):
			plt.plot(fitting_x,single_gaussian(fit_para,fitting_x),color='b',ls='-',lw=3)
			if (DIB_info['fitting_type']==5):
				plt.axvspan(DIB_info['exclude'][0],DIB_info['exclude'][1],color='gray',alpha=0.5,lw=0.0)
		if (DIB_info['fitting_type']==2) | (DIB_info['fitting_type']==7):
			plt.plot(fitting_x,double_gaussian(fit_para,fitting_x),color='b',ls='-',lw=3)
			#f_para = fit_para[0:3]
			#f_para.append(fit_para[-1])
			#plt.plot(fitting_x,single_gaussian(f_para,fitting_x),color='g',ls='--',lw=2)
			#s_para = fit_para[3:]
			#plt.plot(fitting_x,single_gaussian(s_para,fitting_x),color='g',ls='--',lw=2)
			a = fit_para[0:3]
			a.append(fit_para[-1])
			b = fit_para[3:]


			plt.plot(fitting_x,single_gaussian(a,fitting_x),color='r',lw=2.5,ls='-')
			plt.plot(fitting_x,single_gaussian(b,fitting_x),color='LimeGreen',lw=2.5,ls='-')


		if DIB_info['fitting_type']==3:
			plt.plot(fitting_x,three_gaussian(fit_para,fitting_x),color='b',ls='-',lw=2)
			a = fit_para[0:3]
			a.append(fit_para[-1])
			b = fit_para[3:6]
			b.append(fit_para[-1])
			c = fit_para[6:]


			plt.plot(fitting_x,single_gaussian(a,fitting_x),color='r',lw=2.5,ls='-')
			plt.plot(fitting_x,single_gaussian(b,fitting_x),color='LimeGreen',lw=2.5,ls='-')
			plt.plot(fitting_x,single_gaussian(c,fitting_x),color='orange',lw=2.5,ls='-')


		if DIB_info['fitting_type']==4:
			plt.plot(fitting_x,four_gaussian(fit_para,fitting_x),color='b',ls='-',lw=1.5)

		plt.axhline(fit_para[-1],ls='--',color='black')
		plt.xlim(fitting_x[0]-0.3*(fitting_x[-1]-fitting_x[0]),fitting_x[-1]+0.3*(fitting_x[-1]-fitting_x[0]))		

	return DIB_output, display_y

def a_line(p,x):
	a,b = p
	return a*x+b
def residuals_line(p, data):
    x, y,err = data
    r = (y-a_line(p,x))/err
    return r	

def single_gaussian(p,x):
	A, mu, sigma, zerop = p	
	return  -1.0*A * np.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma))+zerop

def residuals_single_g(p,data):
	x, y,err = data
	r = (y-single_gaussian(p,x))/err
	return r		

def double_gaussian(p,x):
	A, mu, sigma, B, mu_b,sigma_b,zerob = p	
	return  -1.0*A * np.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma))-1.0*B * np.exp(-(x-mu_b)*(x-mu_b)/(2.0*sigma_b*sigma_b))+zerob

def residuals_double_g(p, data):
    x, y ,err = data
    r = (y-double_gaussian(p,x))/err
    return r		

def three_gaussian(p,x):
	A, mu, sigma, B,mu_b,sigma_b,C,mu_c,sigma_c,zerop = p	
	return  -1.0*A * np.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma))-1.0*B * np.exp(-(x-mu_b)*(x-mu_b)/(2.0*sigma_b*sigma_b))-1.0*C * np.exp(-(x-mu_c)*(x-mu_c)/(2.0*sigma_c*sigma_c))+zerop

def three_gaussian_special(p,x):
	A, mu, sigma, B,distance_b,sigma_b,C,distance_c,sigma_c,zerop = p	
	return  -1.0*A * np.exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma))-1.0*B * np.exp(-(x-(mu+distance_b))*(x-(mu+distance_b))/(2.0*sigma_b*sigma_b))-1.0*C * np.exp(-(x-(mu+distance_c))*(x-(mu+distance_c))/(2.0*sigma_c*sigma_c))+zerop



def residuals_three_g(p, data):
    x, y,err = data
    r = (y-three_gaussian(p,x))/err
    return r		

def residuals_three_special(p, data):
    x, y,err = data
    r = (y-three_gaussian_special(p,x))/err
    return r		


def save_into_fits(EW_matrix_total,dib_name,output_name,E_BV):
	table = []
	table.append(pf.PrimaryHDU(np.arange(0,100)))	
	col_def = []
	for i_dib in range(0,len(dib_name)):
		flux_col = pf.Column(name=dib_name[i_dib],format='E',array=EW_matrix_total[i_dib,:,0])
		error_col = pf.Column(name=dib_name[i_dib]+'_error',format='E',array=EW_matrix_total[i_dib,:,1])
		center_col = pf.Column(name=dib_name[i_dib]+'_wavelength',format='E',array=EW_matrix_total[i_dib,:,2])
		#m_error_col = pf.Column(name=para_dict[i_dib][5]+'_error_m',format='E',array=EW_matrix_total[i_data,:,i_dib,2])

		col_def.append(flux_col)
		col_def.append(error_col)
		col_def.append(center_col)

		#col_def.append(m_error_col)
	col_def.append(pf.Column(name='E_BV',format='E',array=E_BV))	
	table.append(pf.new_table(pf.ColDefs(col_def)))
	HDUList = pf.HDUList(table)
	HDUList.writeto(output_name,clobber=True)

def give_me_the_list():
	dib_list = ['4430','5781','6284','NaD']

	print "Caution Not the best"
	
	return dib_list
def give_me_dib_list():
	dib_list = give_me_the_list()
	individual_dib_list = []
	for i_dib_list in dib_list:
		dib_info = DIB_fitting_info(i_dib_list)
		N_dib = dib_info['fitting_type']
		if dib_info['fitting_type']==5:
			N_dib=1
		if dib_info['fitting_type']==7:
			N_dib=1

		for i_n_dib in range(0,N_dib):
			individual_dib_list.append(dib_info['dib_name'][i_n_dib])
	return individual_dib_list


def fitting_sky_map(spec,output_name):
	dib_list = give_me_the_list()
	wave = wave = data_info.wave_air()
	individual_dib_list = give_me_dib_list()
	spec_n = len(spec[0].data)
	EW_matrix = np.zeros((len(individual_dib_list),spec_n,3))-999.0
	for i_spec in range(0,spec_n):
		count_dib = 0
		if spec[4].data[i_spec]>5:	
			print i_spec	
			for index,i_dib_list in enumerate(dib_list):
				#plt.subplot(spec_n,len(dib_list),index+1+i_spec*len(dib_list))

				dib_info = DIB_fitting_info(i_dib_list)
				DIB_output,display_y = go_fitting(wave,spec[1].data[i_spec],dib_info,False,-1)
		
				N_dib = dib_info['fitting_type']
				if dib_info['fitting_type']==5:
					N_dib=1
				if dib_info['fitting_type']==7:
					N_dib=1	
					
				for i_n_dib in range(0,N_dib):
					EW_matrix[count_dib,i_spec,0]=DIB_output[i_n_dib]['area'][0]
					EW_matrix[count_dib,i_spec,1]=DIB_output[i_n_dib]['area'][1]
					EW_matrix[count_dib,i_spec,2]=DIB_output[i_n_dib]['fit_para'][1]

					count_dib=count_dib+1

	save_into_fits(EW_matrix,individual_dib_list,output_name,spec[2].data)	

def fitting_sky_map_LRGs(spec,output_name):
	dib_list = give_me_the_list()
	wave = wave = data_info.wave_air()
	individual_dib_list = give_me_dib_list()
	spec_n = len(spec[0].data)
	EW_matrix = np.zeros((len(individual_dib_list),spec_n,3))-999.0
	for i_spec in range(0,spec_n):
		count_dib = 0
		if spec[3].data[i_spec]>5:	
			print i_spec	
			for index,i_dib_list in enumerate(dib_list):
				#plt.subplot(spec_n,len(dib_list),index+1+i_spec*len(dib_list))

				dib_info = DIB_fitting_info(i_dib_list)
				DIB_output,display_y = go_fitting(wave,spec[0].data[i_spec],dib_info,False,-1)
		
				N_dib = dib_info['fitting_type']
				if dib_info['fitting_type']==5:
					N_dib=1
				if dib_info['fitting_type']==7:
					N_dib=1	
					
				for i_n_dib in range(0,N_dib):
					EW_matrix[count_dib,i_spec,0]=DIB_output[i_n_dib]['area'][0]
					EW_matrix[count_dib,i_spec,1]=DIB_output[i_n_dib]['area'][1]
					EW_matrix[count_dib,i_spec,2]=DIB_output[i_n_dib]['fit_para'][1]
					count_dib=count_dib+1

	save_into_fits(EW_matrix,individual_dib_list,output_name,spec[2].data)	



def fitting_EBV(spec,output_name):
	wave = wave = data_info.wave_air()
	dib_list = give_me_the_list()
	individual_dib_list = give_me_dib_list()		
	spec_n = len(spec[0].data)
	print individual_dib_list
	EW_matrix = np.zeros((len(individual_dib_list),spec_n,2))
	para_matrix = np.zeros((len(individual_dib_list),spec_n,3))
	
	for i_spec in range(0,spec_n):
		count_dib = 0
		for index,i_dib_list in enumerate(dib_list):
			#plt.subplot(spec_n,len(dib_list),index+1+i_spec*len(dib_list))

			dib_info = DIB_fitting_info(i_dib_list)
			DIB_output,display_y = go_fitting(wave,spec[1].data[i_spec],dib_info,False,-1)
			
			N_dib = dib_info['fitting_type']
			if (dib_info['fitting_type']==5) | (dib_info['fitting_type']==7):
				N_dib=1
			for i_n_dib in range(0,N_dib):
				EW_matrix[count_dib,i_spec,0]=DIB_output[i_n_dib]['area'][0]
				EW_matrix[count_dib,i_spec,1]=DIB_output[i_n_dib]['area'][1]
				para_matrix[count_dib,i_spec,:]=DIB_output[i_n_dib]['fit_para'][0:3]
				
				count_dib=count_dib+1
				
	save_into_fits(EW_matrix,individual_dib_list,output_name,spec[3].data['E_BV'])
	return para_matrix


def fitting_EBV_with_bootstrap(spec,output_name,method):
	wave = wave = data_info.wave_air()
	dib_list = give_me_the_list()
	individual_dib_list = give_me_dib_list()		
	spec_n = len(spec[0].data)
	print individual_dib_list
	EW_matrix = np.zeros((len(individual_dib_list),spec_n,2))
	for i_spec in range(0,spec_n):
		count_dib = 0
		for index,i_dib_list in enumerate(dib_list):
			#plt.subplot(spec_n,len(dib_list),index+1+i_spec*len(dib_list))

			dib_info = DIB_fitting_info(i_dib_list)
			DIB_output,display_y = go_fitting(wave,spec[1].data[i_spec],dib_info,False,method)
			tmp_dict = {}
			for i_boot in range(0,100):			
				tmp_dict[i_boot] = go_fitting(wave,spec[2].data[i_spec,:,i_boot,1],dib_info,False,method)[0]
			
			
			
			
			
			N_dib = dib_info['fitting_type']
			if (dib_info['fitting_type']==5) | (dib_info['fitting_type']==7):
				N_dib=1
			for i_n_dib in range(0,N_dib):
				EW_matrix[count_dib,i_spec,0]=DIB_output[i_n_dib]['area'][0]
				tmp_area = []
				for i_boot in range(0,100):
					tmp_area.append(tmp_dict[i_boot][i_n_dib]['area'][0])
				
				EW_matrix[count_dib,i_spec,1]=np.std(tmp_area)
				count_dib=count_dib+1

	save_into_fits(EW_matrix,individual_dib_list,output_name,spec[3].data['E_BV'])



def provide_final_spectrum(spectrum):
	
	wave = wave = data_info.wave_air()	
	cut = (wave>4300) & (wave<6700)
	sigma,mean = spec_tools.sigma_clipping(spectrum[cut],3)
	spec_err = np.arange(0,len(wave))*0.0+sigma
	dib_list = give_me_the_list()	
	for index,i_dib_list in enumerate(dib_list):
		DIB_info = DIB_fitting_info(i_dib_list)	
		
		cut_for_line = np.where(((wave>DIB_info['range_for_line']['L'][0]) & (wave<DIB_info['range_for_line']['L'][1])) |\
			((wave>DIB_info['range_for_line']['R'][0]) & (wave<DIB_info['range_for_line']['R'][1])))

		para = [1.0,1.0]
		fitobj = kmpfit.Fitter(residuals=residuals_line,data=(wave[cut_for_line[0]],spectrum[cut_for_line[0]],spec_err[cut_for_line[0]]))
		fitobj.fit(params0=para)
		line_para = fitobj.params
		#-----------------------------------
		#fit gaussian
		cut_for_fitting = np.where((wave>DIB_info['range_for_line']['L'][0]) & (wave<DIB_info['range_for_line']['R'][1]))	
		fitting_x = wave[cut_for_fitting]
		spec_fit = spectrum[cut_for_fitting]
	
		fitting_y = spec_fit/a_line(line_para,fitting_x)
	
		fitting_yerr = spec_err[cut_for_fitting]
		display_y = spectrum
		display_y[cut_for_fitting]=fitting_y


	return display_y




def fitting_one_spectrum(spectrum):
	wave = wave = data_info.wave_air()
	dib_list = give_me_the_list()
	individual_dib_list = give_me_dib_list()		
	display_y = spectrum	
	count_dib = 0
	for index,i_dib_list in enumerate(dib_list):
		plt.subplot(1,4,index+1)
		para = {'setting':1,'fontsize':8,'lw':1}
		if index ==0:
			para['setting']=0
		my_set_up.make_plot_set_up(para)
		dib_info = DIB_fitting_info(i_dib_list)
		DIB_output,output_spec = go_fitting(wave,display_y,dib_info,True,-1)
		display_y=output_spec
		#plt.ylim(0.995,1.0015)
	return display_y


def fitting_one_spectrum_para(spectrum):
	wave = wave = data_info.wave_air()
	dib_list = give_me_the_list()
	individual_dib_list = give_me_dib_list()		
	display_y = spectrum	
	count_dib = 0
	for index,i_dib_list in enumerate(dib_list):
		#plt.subplot(1,15,index+1)
		#para = {'setting':1,'fontsize':8,'lw':1}
		#if index ==0:
		#	para['setting']=0
		#my_set_up.make_plot_set_up(para)
		dib_info = DIB_fitting_info(i_dib_list)
		DIB_output,output_spec = go_fitting(wave,display_y,dib_info,False,True)
		display_y=output_spec
		#plt.ylim(0.995,1.0015)
	return 



def fitting_n_spectra(spectra):
	wave = wave = data_info.wave_air()
	dib_list = give_me_the_list()
	individual_dib_list = give_me_dib_list()		
	count_dib = 0
	color=['r','g','orange']
	for i_spectra in range(0,len(spectra)):
		for index,i_dib_list in enumerate(dib_list):
			plt.subplot(3,5,index+1+15*i_spectra)
			para = {'setting':1,'fontsize':13,'lw':1.5}
			if index % 5==0:
				para['setting']=0
			my_set_up.make_plot_set_up(para)
			dib_info = DIB_fitting_info(i_dib_list)
			DIB_output = go_fitting(wave,spectra[i_spectra],dib_info,True,-1)
			print dib_info['range_for_line']
			plt.ylim(0.983-0.002*i_spectra,1.00351+0.001*i_spectra)
			n_dib = dib_info['fitting_type']
			if (dib_info['fitting_type']==5) | (dib_info['fitting_type']==7):
				n_dib=1
			
			for i_type in range(0,n_dib):
				if (dib_info['fitting_type']==2) | (dib_info['fitting_type']==3):
					plt.text(dib_info['range_for_line']['L'][0]-0.15*(dib_info['range_for_line']['R'][0]-dib_info['range_for_line']['L'][0]),0.991-0.003*i_type,individual_dib_list[count_dib],color=color[i_type],fontsize=15)
				else:
					plt.text(dib_info['range_for_line']['L'][0]-0.20*(dib_info['range_for_line']['R'][0]-dib_info['range_for_line']['L'][0]),0.985-0.002*i_type,individual_dib_list[count_dib],color='b',fontsize=15)
					
				count_dib=count_dib+1

def display_fitting():
	#spec = pf.open('../a_compile_spectra_and_catalogs/140604_EBV_f_h2_lt_035.fits')
	#spec = pf.open('/Volumes/gwln1/projects/DIBs/code/diffuse_interstellar_bands-/trunk/a_compile_spectra_and_catalogs/140610_EBV_problematic_removed_bootstrap.fits')
	spec = pf.open('/Volumes/gwln1/projects/DIBs/code/diffuse_interstellar_bands-/trunk/a_compile_spectra_and_catalogs/140604_all_E_BV_GT_010.fits')
	spectrum = spec[0].data[0,:]
	plt.figure(figsize=(18,7.5))
	plt.subplots_adjust(wspace=0.1,hspace=0.2,right=0.95,top=0.95,left=0.08,bottom=0.10)

	dib = pf.getdata('../../../../data/DIBs_new.fit',1)
	mask_c = mask.give_me_mask(dib)
	#mask_c = np.arange(0.,3955)*0.0+1.0
	median_filter = mask.run_mask_median_filter_singe(spectrum,mask_c,25)

	#fitting_one_spectrum(spectrum)
	spectra = np.zeros((1,len(spectrum)))
	for i in range(0,1):
		spectra[i,:] = spec[0].data[0,:]/median_filter
	
	#print spec[3].data[0]	
	fitting_n_spectra(spectra)
	plt.subplot(3,5,13)
	plt.xlabel('wavelength [$\\rm \\AA$]',fontsize=20)
	plt.subplot(3,5,6)
	plt.ylabel('Normalized flux',fontsize=20)


def fitting_5780_region_spectra(spectra):
	wave = wave = data_info.wave_air()
	name = '5850'	
	plt.figure(figsize=(5.5,5))		
	plt.subplots_adjust(left=0.15,bottom=0.12)
	plt.subplot(1,1,1)
	para = {'setting':0,'fontsize':15,'lw':1.5}

	my_set_up.make_plot_set_up(para)
	dib_info = DIB_fitting_info(name)
	DIB_output = go_fitting(wave,spectra,dib_info,True,2)
	print DIB_output[0][0]['area'][0],DIB_output[0][0]['area'][1]
	print DIB_output[0][1]['area'][0],DIB_output[0][1]['area'][1]
	print DIB_output[0][2]['area'][0],DIB_output[0][2]['area'][1]

	a = DIB_output[0][0]['fit_para'][0:3]
	a.append(DIB_output[0][0]['zero'])
	b = DIB_output[0][1]['fit_para'][0:3]
	b.append(DIB_output[0][0]['zero'])
	c = DIB_output[0][2]['fit_para'][0:3]
	c.append(DIB_output[0][0]['zero'])


	plt.plot(wave,single_gaussian(a,wave),color='b',lw=1.5)
	plt.plot(wave,single_gaussian(b,wave),color='LimeGreen',lw=1.5)
	plt.plot(wave,single_gaussian(c,wave),color='orange',lw=1.5)
	plt.axhline(DIB_output[0][0]['zero'],ls='--',color='black',lw=2)
	plt.xlim(5755,5819)
	plt.ylim(0.98,1.00251+0.001)
	plt.text(5805,0.987,'5778',color='b',fontsize=15)
	plt.text(5805,0.985,'5780',color='g',fontsize=15)
	plt.text(5805,0.983,'5797',color='orange',fontsize=15)
	plt.text(5805,0.981,'Total',color='red',fontsize=15)
	plt.ylabel('Normalized flux',fontsize=15)
	plt.xlabel('wavelength [$\\rm \\AA$]',fontsize=15)


def redundenct():
			
	if fixed_width == 2:
		fitobj.parinfo = ({'limits':(-100.0,1.0)},\
			{'limits':(para[1]-0.5,para[1]+0.5)},
			{'fixed':True},\
			
			{'limits':(-100.0,1.0)},\
			{'fixed':True},\
			{'fixed':True},\
			
			{'limits':(-100.0,1.0)},\
			{'fixed':True},\
			{'fixed':True},\
			{'fixed':True})
		para = [DIB_info['para'][0],DIB_info['para'][1],DIB_info['para'][2],DIB_info['para'][3],DIB_info['para'][4]-DIB_info['para'][1],DIB_info['para'][5],\
			DIB_info['para'][6],DIB_info['para'][7]-DIB_info['para'][1],DIB_info['para'][8],DIB_info['para'][9]]	
		fitobj = kmpfit.Fitter(residuals=residuals_three_special,data=(fitting_x,fitting_y,fitting_yerr))
				


def fitting_radius(spec,wave,E_BV,output_name):
	#wave = wave = data_info.wave_air()
	dib_list = give_me_the_list()
	individual_dib_list = give_me_dib_list()		
	spec_n = len(spec)
	print individual_dib_list
	EW_matrix = np.zeros((len(individual_dib_list),spec_n,2))
	para_matrix = np.zeros((len(individual_dib_list),spec_n,3))
	
	for i_spec in range(0,spec_n):
		count_dib = 0
		for index,i_dib_list in enumerate(dib_list):
			#plt.subplot(spec_n,len(dib_list),index+1+i_spec*len(dib_list))

			dib_info = DIB_fitting_info(i_dib_list)
			DIB_output,display_y = go_fitting(wave,spec[i_spec],dib_info,False,0)
			
			N_dib = dib_info['fitting_type']
			if (dib_info['fitting_type']==5) | (dib_info['fitting_type']==7):
				N_dib=1
			for i_n_dib in range(0,N_dib):
				EW_matrix[count_dib,i_spec,0]=DIB_output[i_n_dib]['area'][0]
				EW_matrix[count_dib,i_spec,1]=DIB_output[i_n_dib]['area'][1]
				para_matrix[count_dib,i_spec,:]=DIB_output[i_n_dib]['fit_para'][0:3]
				
				count_dib=count_dib+1
				
	save_into_fits(EW_matrix,individual_dib_list,output_name,E_BV)
	return para_matrix


#spec = pf.open('/Volumes/gwln1/projects/DIBs/code/diffuse_interstellar_bands-/trunk/a_compile_spectra_and_catalogs/140604_all_E_BV_GT_005.fits')
#fitting_5780_region_spectra(spec[1].data[0])
#plt.ioff()
#display_fitting()
#plt.savefig('Example_of_fitting.pdf')

#plt.subplot(3,5,13)
#plt.axvspan(6300,6310,color='gray',lw=0.0,alpha=0.5)
#fitting_EBV()


#fitting_H_2_fraction()
#fitting_sky_map()