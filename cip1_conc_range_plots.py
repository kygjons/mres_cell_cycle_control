#!/usr/bin/env python3

#Initialization of coding environment and loading of appropriate packages
import numpy as np                      
import math                               
from scipy.integrate import odeint                        
import matplotlib.pyplot as plt  

#Function for Cip1 Screening Model
def cell_cycle(y, t):
    # Tracked Species
    Clb12, Clb56, Sic1, Clb12_Sic1, Clb56_Sic1, Clb34, Clb34_Sic1, Cln2, Cip1, Cip1_Cln2 = y
    
    # Model (ODEs and Equations)
    
    dClb12dt = k_exp_12*(((Clb12)/(1+Clb12))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))) - k_dil_12*Clb12 - (k_a*Clb12*Sic1) + (k_d*Clb12_Sic1) + k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb12_Sic1 - k_deg_Clb*Clb12*((Clb12)/(1+Clb12))
    dClb56dt = k_exp_56*(((1)/(1+(Clb56)))) - k_dil_56*Clb56 - (k_a*Clb56*Sic1) + (k_d*Clb56_Sic1) + k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb56_Sic1 - k_deg_Clb*Clb56*(((Clb12)/(1+Clb12))+((Clb34)/(1+Clb34)))
    dSic1dt = k_exp_Sic1*(((1)/(1+(Clb12)))+((1)/(1+(Clb34)))+((1)/(1+(Clb56)))) - k_dil_Sic1*Sic1 - (k_a*Clb12*Sic1) + (k_d*Clb12_Sic1) - (k_a*Clb56*Sic1) + (k_d*Clb56_Sic1) - (k_a*Clb34*Sic1) + (k_d*Clb34_Sic1) - k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Sic1
    dClb34dt = k_exp_34*(((Clb56)/(1+Clb56))) - k_dil_34*Clb34 - (k_a*Clb34*Sic1) + (k_d*Clb34_Sic1) + k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb34_Sic1 - k_deg_Clb*Clb34*((Clb12)/(1+Clb12))
    
    d_Clb12_Sic1dt = (k_a*Clb12*Sic1) - (k_d*Clb12_Sic1) - k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb12_Sic1 - k_dil*Clb12_Sic1
    d_Clb56_Sic1dt = (k_a*Clb56*Sic1) - (k_d*Clb56_Sic1) - k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb56_Sic1 - k_dil*Clb56_Sic1
    d_Clb34_Sic1dt = (k_a*Clb34*Sic1) - (k_d*Clb34_Sic1) - k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb34_Sic1 - k_dil*Clb34_Sic1
    
    dCln2dt = k_exp_Cln2*(((Cln2)/(1+(Cln2)))+((1)/(1+(Clb12)))+((1)/(1+(Clb56)))) - k_dil_Cln2*Cln2 - k_deg_Cln*Cln2*((Cln2)/(1+Cln2)+(Clb12)/(1+Clb12)+(Clb34)/(1+Clb34)+(Clb56)/(1+Clb56)) - k_a*Cip1*Cln2 + k_d*Cip1_Cln2 + k_deg*Cip1_Cln2*(((Clb56)/(1+(Clb56))))
    
    dCip1dt = k_induction_Cip1 + k_exp_Cip1*(((Cln2)/(1+(Cln2)))+((1)/(1+(Clb56)))) - k_dil_Cip1*Cip1 - k_a*Cip1*Cln2 + k_d*Cip1_Cln2 - k_deg*Cip1*(((Clb56)/(1+(Clb56))))
    dCip1_Cln2dt = k_a*Cip1*Cln2 - k_d*Cip1_Cln2 - k_dil*Cip1_Cln2 - k_deg*Cip1_Cln2*(((Clb56)/(1+(Clb56))))
    
    solutions = [dClb12dt, dClb56dt, dSic1dt, d_Clb12_Sic1dt, d_Clb56_Sic1dt, dClb34dt, d_Clb34_Sic1dt, dCln2dt, dCip1dt, dCip1_Cln2dt]
    
    return solutions

#Parameters for Cip1 Screening Model
k_deg = 1 #in min^-1
k_a =  50 #in min^-1
k_d =  0.05 #in min^-1
k_deg_Clb =  0.4 #in min^-1 or 0.4 
k_deg_Cln = 5 #in min^-1

k_exp_Sic1= 0.1 #in uM min^-1
k_dil_Sic1 = 0.01 #in min^-1

k_exp_12 =  0.05 #in min^-1
k_dil_12 =  0.03 #in min^-1

k_exp_34 = 0.03 #in min^-1
k_dil_34 = 0.03 #in min^-1

k_exp_56 = 0.012 #in min^-1
k_dil_56 = 0.01 #in min^-1

k_exp_Cln2 = 0.15 #in min^-1
k_dil_Cln2 = 0.12 #in min^-1

k_dil=0.01

k_exp_Cip1 = 0.02
k_dil_Cip1 = 0.02

#Array for k_induction_Cip1 values to be simulated
parameter_range_k_induction_Cip1 = [0, 0.05, 0.1, 0.2, 0.3, 0.5]

#Creating empty arrays for each protein species
Clb12_arrays_k_exp_Cip1 = []
Clb34_arrays_k_exp_Cip1 = []
Clb56_arrays_k_exp_Cip1 = []
Sic1_arrays_k_exp_Cip1 = []
Clb12_Sic1_arrays_k_exp_Cip1 = []
Clb34_Sic1_arrays_k_exp_Cip1 = []
Clb56_Sic1_arrays_k_exp_Cip1 = []
Cln2_arrays_k_exp_Cip1 = []
Cip1_arrays_k_exp_Cip1 = []
Cip1_Cln2_arrays_k_exp_Cip1 = []

#For loop to run multiple simulations with a different k_exp_Cip1 value each time
for i in parameter_range_k_induction_Cip1:
    # Setting k_induction_Cip1 value for run
    k_induction_Cip1 = i 
    
    #Setting initial conditions
    y0_model = [0.1375718008479316, 0.1594779194006006, 0.017081417269881664, 0.22793430826192018, 0.2739601812180509, 0.042811338257718164, 0.071643195723027, 0.12544390359068047, 0.1, 0.1]
    #Setting time interval for integration
    t_interval = np.linspace(0, 250, 5000)
    #Integrating ODEs
    sol = odeint(func = cell_cycle, y0 = y0_model, t = t_interval)
    sol.transpose()
    
    # Assigning each array to the corresponding protein species array
    Clb12_arrays_k_exp_Cip1.append(sol.transpose()[0])
    Clb34_arrays_k_exp_Cip1.append(sol.transpose()[5])
    Clb56_arrays_k_exp_Cip1.append(sol.transpose()[1])
    Sic1_arrays_k_exp_Cip1.append(sol.transpose()[2])
    Clb12_Sic1_arrays_k_exp_Cip1.append(sol.transpose()[3])
    Clb34_Sic1_arrays_k_exp_Cip1.append(sol.transpose()[6])
    Clb56_Sic1_arrays_k_exp_Cip1.append(sol.transpose()[4])
    Cln2_arrays_k_exp_Cip1.append(sol.transpose()[7])
    Cip1_arrays_k_exp_Cip1.append(sol.transpose()[8])
    Cip1_Cln2_arrays_k_exp_Cip1.append(sol.transpose()[9])

#Creating empty arrays for total protein calculations
Clb12_T_k_exp_Cip1 = []
Clb34_T_k_exp_Cip1 = []
Clb56_T_k_exp_Cip1 = []
Sic1_T_k_exp_Cip1 = []
Cln2_T_k_exp_Cip1 = []

#Calculating Total Protein In and Out of Complex for Each Protein Species
for i in np.arange(0,6,1):
    Clb12_T_k_exp_Cip1.append(np.add(Clb12_arrays_k_exp_Cip1[i], Clb12_Sic1_arrays_k_exp_Cip1[i]))
    Clb34_T_k_exp_Cip1.append(np.add(Clb34_arrays_k_exp_Cip1[i], Clb34_Sic1_arrays_k_exp_Cip1[i]))
    Clb56_T_k_exp_Cip1.append(np.add(Clb56_arrays_k_exp_Cip1[i], Clb56_Sic1_arrays_k_exp_Cip1[i]))
    Sic1_12_34_k_exp_Cip1 = np.add(Clb12_Sic1_arrays_k_exp_Cip1[i], Clb34_Sic1_arrays_k_exp_Cip1[i])
    Sic1_12_34_56_k_exp_Cip1 = np.add(Sic1_12_34_k_exp_Cip1, Clb56_Sic1_arrays_k_exp_Cip1[i])
    Sic1_T_k_exp_Cip1.append(np.add(Sic1_12_34_56_k_exp_Cip1, Sic1_arrays_k_exp_Cip1[i]))
    Cln2_T_k_exp_Cip1.append(np.add(Cln2_arrays_k_exp_Cip1[i], Cip1_Cln2_arrays_k_exp_Cip1[i]))

#Creating empty arrays for scaled values
Clb12_scaled_k_exp_Cip1 = []
Clb34_scaled_k_exp_Cip1 = []
Clb56_scaled_k_exp_Cip1 = []
Sic1_scaled_k_exp_Cip1 = []
Cln2_scaled_k_exp_Cip1 = []

#Scaling to percentage of maximum
for i in np.arange(0,6,1):
    Clb12_scaled_k_exp_Cip1.append((Clb12_T_k_exp_Cip1[i]/np.amax(Clb12_T_k_exp_Cip1[i]))*100)
    Clb34_scaled_k_exp_Cip1.append((Clb34_T_k_exp_Cip1[i]/np.amax(Clb34_T_k_exp_Cip1[i]))*100)
    Clb56_scaled_k_exp_Cip1.append((Clb56_T_k_exp_Cip1[i]/np.amax(Clb56_T_k_exp_Cip1[i]))*100)
    Sic1_scaled_k_exp_Cip1.append((Sic1_T_k_exp_Cip1[i]/np.amax(Sic1_T_k_exp_Cip1[i]))*100)
    Cln2_scaled_k_exp_Cip1.append((Cln2_T_k_exp_Cip1[i]/np.amax(Cln2_T_k_exp_Cip1[i]))*100)    

#Creating figure with 6 subplots split over two rows
fig, sa_k_exp_Cip1 = plt.subplots(2, 3, figsize = (15, 7.5), sharey=False, tight_layout=True)
#Creating arrays to assign values for each subplot
plots_row = [0, 0, 0, 1, 1, 1]
plots_col = [0, 1, 2, 0, 1, 2]
#For loop to plot corresponding values in each subplot
for i in np.arange(0,6,1):
    sa_k_exp_Cip1[plots_row[i], plots_col[i]].plot(t_interval, Clb12_scaled_k_exp_Cip1[i], label = "Clb1,2")
    sa_k_exp_Cip1[plots_row[i], plots_col[i]].plot(t_interval, Clb34_scaled_k_exp_Cip1[i], label = "Clb3,4")
    sa_k_exp_Cip1[plots_row[i], plots_col[i]].plot(t_interval, Clb56_scaled_k_exp_Cip1[i], label = "Clb5,6")
    sa_k_exp_Cip1[plots_row[i], plots_col[i]].plot(t_interval, Sic1_scaled_k_exp_Cip1[i], label = "Sic1")
    sa_k_exp_Cip1[plots_row[i], plots_col[i]].set_xlim(0, 250)

#Setting plot titles and shading each G1 phase
sa_k_exp_Cip1[0, 0].axvspan(t_interval[2250], t_interval[3150], facecolor="lightcoral")
sa_k_exp_Cip1[0, 0].set_title("$k_{induction_ Cip1}$ = "+str(parameter_range_k_induction_Cip1[0])+" $min^{-1}$", fontsize=15)

sa_k_exp_Cip1[0, 1].axvspan(t_interval[2400], t_interval[3400], facecolor="lightcoral")
sa_k_exp_Cip1[0, 1].set_title("$k_{induction_ Cip1}$ = "+str(parameter_range_k_induction_Cip1[1])+" $min^{-1}$", fontsize=15)

sa_k_exp_Cip1[0, 2].axvspan(t_interval[2600], t_interval[3675], facecolor="lightcoral")
sa_k_exp_Cip1[0, 2].set_title("$k_{induction_ Cip1}$ = "+str(parameter_range_k_induction_Cip1[2])+" $min^{-1}$", fontsize=15)

sa_k_exp_Cip1[1, 0].axvspan(t_interval[325], t_interval[2500], facecolor="lightcoral")
sa_k_exp_Cip1[1, 0].set_title("$k_{induction_ Cip1}$ = "+str(parameter_range_k_induction_Cip1[3])+" $min^{-1}$", fontsize=15)

sa_k_exp_Cip1[1, 1].set_title("$k_{induction_ Cip1}$ = "+str(parameter_range_k_induction_Cip1[4])+" $min^{-1}$", fontsize=15)

sa_k_exp_Cip1[1, 2].set_title("$k_{induction_ Cip1}$ = "+str(parameter_range_k_induction_Cip1[5])+" $min^{-1}$", fontsize=15)

#Setting axes labels
for i in range(0, 2, 1):
    sa_k_exp_Cip1[i, 1].set_xlabel("Time (min)", fontsize=15)
for i in range(0, 2, 1):
    sa_k_exp_Cip1[i, 0].set_ylabel("Concentration (% of Maximum)", fontsize=15)
    
plt.show()