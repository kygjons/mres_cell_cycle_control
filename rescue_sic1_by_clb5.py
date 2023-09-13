#!/usr/bin/env python3

#Initialization of coding environment and loading of appropriate packages
import numpy as np                      
import math                               
from scipy.integrate import odeint                        
import matplotlib.pyplot as plt  

#Function for Baseline Model
def cell_cycle(y, t):
    # Tracked Species
    Clb12, Clb56, Sic1, Clb12_Sic1, Clb56_Sic1, Clb34, Clb34_Sic1, Cln2 = y
    
    # Model (ODEs and Equations)
    
    dClb12dt = k_exp_12*(((Clb12)/(1+Clb12))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))) - k_dil_12*Clb12 - (k_a*Clb12*Sic1) + (k_d*Clb12_Sic1) + k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb12_Sic1 - k_deg_Clb*Clb12*((Clb12)/(1+Clb12))
    dClb56dt = k_exp_56*(((1)/(1+(Clb56)))) - k_dil_56*Clb56 - (k_a*Clb56*Sic1) + (k_d*Clb56_Sic1) + k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb56_Sic1 - k_deg_Clb*Clb56*(((Clb12)/(1+Clb12))+((Clb34)/(1+Clb34)))
    dSic1dt = k_exp_Sic1*(((1)/(1+(Clb12)))+((1)/(1+(Clb34)))+((1)/(1+(Clb56)))) - k_dil_Sic1*Sic1 - (k_a*Clb12*Sic1) + (k_d*Clb12_Sic1) - (k_a*Clb56*Sic1) + (k_d*Clb56_Sic1) - (k_a*Clb34*Sic1) + (k_d*Clb34_Sic1) - k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Sic1
    dClb34dt = k_exp_34*(((Clb56)/(1+Clb56))) - k_dil_34*Clb34 - (k_a*Clb34*Sic1) + (k_d*Clb34_Sic1) + k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb34_Sic1 - k_deg_Clb*Clb34*((Clb12)/(1+Clb12))
    
    d_Clb12_Sic1dt = (k_a*Clb12*Sic1) - (k_d*Clb12_Sic1) - k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb12_Sic1 - k_dil*Clb12_Sic1
    d_Clb56_Sic1dt = (k_a*Clb56*Sic1) - (k_d*Clb56_Sic1) - k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb56_Sic1 - k_dil*Clb56_Sic1
    d_Clb34_Sic1dt = (k_a*Clb34*Sic1) - (k_d*Clb34_Sic1) - k_deg*(((Clb12)/(1+(Clb12)))+((Clb56)/(1+(Clb56)))+((Clb34)/(1+(Clb34)))+((Cln2)/(1+(Cln2))))*Clb34_Sic1 - k_dil*Clb34_Sic1
    
    dCln2dt = k_exp_Cln2*(((Cln2)/(1+(Cln2)))+((1)/(1+(Clb12)))+((1)/(1+(Clb56)))) - k_dil_Cln2*Cln2 - k_deg_Cln*Cln2*((Cln2)/(1+Cln2)+(Clb12)/(1+Clb12)+(Clb34)/(1+Clb34)+(Clb56)/(1+Clb56))
    
    solutions = [dClb12dt, dClb56dt, dSic1dt, d_Clb12_Sic1dt, d_Clb56_Sic1dt, dClb34dt, d_Clb34_Sic1dt, dCln2dt]
    
    return solutions

#Assigning parameter values for paramaters that remain unchanged by mutation
k_deg = 1 #in min^-1
k_a =  50 #in min^-1
k_d =  0.05 #in min^-1
k_deg_Clb =  0.4 #in min^-1 or 0.4 
k_deg_Cln = 5 #in min^-1

k_dil_Sic1 = 0.01 #in min^-1

k_exp_12 =  0.05 #in min^-1
k_dil_12 =  0.03 #in min^-1

k_exp_34 = 0.03 #in min^-1
k_dil_34 = 0.03 #in min^-1

k_dil_56 = 0.01 #in min^-1

k_exp_Cln2 = 0.15 #in min^-1
k_dil_Cln2 = 0.12 #in min^-1

k_dil=0.01

#Assigning parameter values for parameters that change with mutation
parameter_range_k_exp_Sic1 = [0.1, 0.08, 0.08]
parameter_range_k_exp_Clb56 = [0.012, 0.012, 0.008]

#Creating arrays to assign each protein species
Clb12_arrays_k_exp_rescue = []
Clb34_arrays_k_exp_rescue = []
Clb56_arrays_k_exp_rescue = []
Sic1_arrays_k_exp_rescue = []
Clb12_Sic1_arrays_k_exp_rescue = []
Clb34_Sic1_arrays_k_exp_rescue = []
Clb56_Sic1_arrays_k_exp_rescue = []
Cln2_arrays_k_exp_rescue = []

#For loop that will run the baseline model with 3 different pairs of k_exp_Sic1 and k_exp_56
for i in np.arange(0,3,1):
    #Setting parameter values for run
    k_exp_Sic1 = parameter_range_k_exp_Sic1[i]
    k_exp_56 = parameter_range_k_exp_Clb56[i]
    
    #Running the Cell Cycle Model
    #Setting the initial conditions
    y0_model = [0.1375718008479316, 0.1594779194006006, 0.017081417269881664, 0.22793430826192018, 0.2739601812180509, 0.042811338257718164, 0.071643195723027, 0.12544390359068047]
    #Setting time interval for integration
    t_interval = np.linspace(0, 250, 5000)
    #Integrating ODEs
    sol = odeint(func = cell_cycle, y0 = y0_model, t = t_interval)
    sol.transpose()
    
    # Assigning each array to the corresponding protein species array
    Clb12_arrays_k_exp_rescue.append(sol.transpose()[0])
    Clb34_arrays_k_exp_rescue.append(sol.transpose()[5])
    Clb56_arrays_k_exp_rescue.append(sol.transpose()[1])
    Sic1_arrays_k_exp_rescue.append(sol.transpose()[2])
    Clb12_Sic1_arrays_k_exp_rescue.append(sol.transpose()[3])
    Clb34_Sic1_arrays_k_exp_rescue.append(sol.transpose()[6])
    Clb56_Sic1_arrays_k_exp_rescue.append(sol.transpose()[4])
    Cln2_arrays_k_exp_rescue.append(sol.transpose()[7])

#Creating empty arrays for total species calcultion
Clb12_T_k_exp_rescue = []
Clb34_T_k_exp_rescue = []
Clb56_T_k_exp_rescue = []
Sic1_T_k_exp_rescue = []

#Calculating total protein in and out of complex for each protein species
#For loop to do this for each of the three runs
for i in np.arange(0,3,1):
    Clb12_T_k_exp_rescue.append(np.add(Clb12_arrays_k_exp_rescue[i], Clb12_Sic1_arrays_k_exp_rescue[i]))
    Clb34_T_k_exp_rescue.append(np.add(Clb34_arrays_k_exp_rescue[i], Clb34_Sic1_arrays_k_exp_rescue[i]))
    Clb56_T_k_exp_rescue.append(np.add(Clb56_arrays_k_exp_rescue[i], Clb56_Sic1_arrays_k_exp_rescue[i]))
    Sic1_12_34_k_exp_rescue = np.add(Clb12_Sic1_arrays_k_exp_rescue[i], Clb34_Sic1_arrays_k_exp_rescue[i])
    Sic1_12_34_56_k_exp_rescue = np.add(Sic1_12_34_k_exp_rescue, Clb56_Sic1_arrays_k_exp_rescue[i])
    Sic1_T_k_exp_rescue.append(np.add(Sic1_12_34_56_k_exp_rescue, Sic1_arrays_k_exp_rescue[i]))

#Creating empty arrays for scaled values
Clb12_scaled_k_exp_rescue = []
Clb34_scaled_k_exp_rescue = []
Clb56_scaled_k_exp_rescue = []
Sic1_scaled_k_exp_rescue = []
Cln2_scaled_k_exp_rescue = []

#Scaling to percentage of maximum
#For loop to do the scaling for each of the three runs
for i in np.arange(0,3,1):
    Clb12_scaled_k_exp_rescue.append((Clb12_T_k_exp_rescue[i]/np.amax(Clb12_T_k_exp_rescue[i]))*100)
    Clb34_scaled_k_exp_rescue.append((Clb34_T_k_exp_rescue[i]/np.amax(Clb34_T_k_exp_rescue[i]))*100)
    Clb56_scaled_k_exp_rescue.append((Clb56_T_k_exp_rescue[i]/np.amax(Clb56_T_k_exp_rescue[i]))*100)
    Sic1_scaled_k_exp_rescue.append((Sic1_T_k_exp_rescue[i]/np.amax(Sic1_T_k_exp_rescue[i]))*100)
    Cln2_scaled_k_exp_rescue.append((Cln2_arrays_k_exp_rescue[i]/np.amax(Cln2_arrays_k_exp_rescue[i]))*100)

#Plotting the values
#Creating an empty figure with three subplots
fig, rescue_Sic1_mutant  = plt.subplots(1, 3, figsize = (17.5, 5), sharey=False, tight_layout=True)
#Creating an array to assign each subplot
plots_col = [0, 1, 2]
#For loop to plot each subplot
for i in np.arange(0,3,1):
    rescue_Sic1_mutant[plots_col[i]].plot(t_interval, Clb12_scaled_k_exp_rescue[i], label = "Clb1,2")
    rescue_Sic1_mutant[plots_col[i]].plot(t_interval, Clb34_scaled_k_exp_rescue[i], label = "Clb3,4")
    rescue_Sic1_mutant[plots_col[i]].plot(t_interval, Clb56_scaled_k_exp_rescue[i], label = "Clb5,6")
    rescue_Sic1_mutant[plots_col[i]].plot(t_interval, Sic1_scaled_k_exp_rescue[i], label = "Sic1")
    rescue_Sic1_mutant[plots_col[i]].legend(fontsize=8, loc="lower right")
    rescue_Sic1_mutant[plots_col[i]].set_xlim(0,250)

#Setting axes titles
rescue_Sic1_mutant[1].set_xlabel("Time (min)", fontsize=20)
rescue_Sic1_mutant[0].set_ylabel("Concentration (% of Maximum)", fontsize=20)

plt.show()