#!/usr/bin/env python3

# Initialization of Coding Environment and Loading of Appropriate Packages
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

# Parameters for Cip1 Screening Model
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
k_induction_Cip1 = 0

#Setting the initial conditions
y0_model = [0.1375718008479316, 0.1594779194006006, 0.017081417269881664, 0.22793430826192018, 0.2739601812180509, 0.042811338257718164, 0.071643195723027, 0.12544390359068047, 0.1, 0.1]

#Setting the time interval for integration
t_interval = np.linspace(0, 300, 5000)

#Integrating the ODEs
sol = odeint(func = cell_cycle, y0 = y0_model, t = t_interval)
sol.transpose()

#Assigning arrays to respective protein species
Clb12_sol = sol.transpose()[0]
Clb56_sol = sol.transpose()[1]
Sic1_sol = sol.transpose()[2]
Clb12_Sic1_sol = sol.transpose()[3]
Clb56_Sic1_sol = sol.transpose()[4]
Clb34_sol = sol.transpose()[5]
Clb34_Sic1_sol = sol.transpose()[6]
Cln2_sol = sol.transpose()[7]
Cip1_sol = sol.transpose()[8]
Cip1_Cln2_sol = sol.transpose()[9]

#Calculating total proteinof each species in and out of complex
Clb12_T = np.add(Clb12_sol, Clb12_Sic1_sol)
Clb34_T = np.add(Clb34_sol, Clb34_Sic1_sol)
Clb56_T = np.add(Clb56_sol, Clb56_Sic1_sol)
Sic1_12_34 = np.add(Clb34_Sic1_sol, Clb56_Sic1_sol)
Sic1_12_34_56 = np.add(Sic1_12_34, Clb12_Sic1_sol)
Sic1_T = np.add(Sic1_sol, Sic1_12_34_56)
Cln2_T = np.add(Cln2_sol, Cip1_Cln2_sol)

#Scaling values to percentage of maximum
Clb12_scaled = (Clb12_T/np.amax(Clb12_T))*100
Clb34_scaled = (Clb34_T/np.amax(Clb34_T))*100
Clb56_scaled = (Clb56_T/np.amax(Clb56_T))*100
Sic1_scaled = (Sic1_T/np.amax(Sic1_T))*100
Cln2_scaled = (Cln2_T/np.amax(Cln2_T))*100

#Plotting the values
fig, cell_cycle_toy = plt.subplots(1, 1, figsize = (10, 5))
cell_cycle_toy.plot(t_interval, Clb12_scaled, label = "Clb1,2")
cell_cycle_toy.plot(t_interval, Clb34_scaled, label = "Clb3,4")
cell_cycle_toy.plot(t_interval, Clb56_scaled, label = "Clb5,6", color="tab:green")
cell_cycle_toy.plot(t_interval, Sic1_scaled, label = "Sic1", color="tab:red")
cell_cycle_toy.plot(t_interval, Cln2_scaled, label = "Cln2", color="tab:purple")

#Modifiying axes labels
cell_cycle_toy.set_xlabel("Time (min)")
cell_cycle_toy.set_ylabel("Concentration (% of Max)")
cell_cycle_toy.set_xlim(0, 300)

#Creating legend for plot
line = []
label = []
handles, labels = cell_cycle_toy.get_legend_handles_labels()
line.extend(handles)
label.extend(labels)
cell_cycle_toy.legend(line, label)

plt.show()