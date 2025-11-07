# Clear all variables when running entire code:
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
# Packages needed
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
# Close all plots when running entire code:
plt.close('all')
# Set default font size in plots:
plt.rcParams.update({'font.size': 12})
import os # For saving plots

# Test #

########## User's settings ##########
n_heights = 3            # Number of layers along reactor height
regions = ["core", "annulus"]  # Division on the horizontal plane
s_species = ["Fe", "FeO", "Fe2O3", "Fe3O4"]   # Different solids species in reactor
# Get the index for strings for better calling
core = regions.index("core")
annulus = regions.index("annulus")
Fe = s_species.index("Fe")
FeO = s_species.index("FeO")
Fe2O3 = s_species.index("Fe2O3")
Fe3O4 = s_species.index("Fe3O4")

########## Parameter Assumption ##########
phi = 1           # Solids particle sphericity
K = 0.2         # Coefficient for lateral mass flow, from core to annulus


########## Fixed Parameter settings ##########
g = 9.8           # Gravity acceleration (m)
# Geometry
bed_height = 20.5     # Total height (m)

# Fluid and solids properties
d_s = 3.5e-5      # diameter of solids particles (m)
rho_s = 7874      # Density of solids particles (kg/m³)
rho_f = 1.205     # Density of fluid gas---air 20 Celsius 1 atom (kg/m3)
mu_f = 18.2e-6    # Dynamic viscosity of fluid gas (Pa.s)
nu_f = mu_f / rho_f # Kinematic viscosity of fluid gas (m2/s)

# Dimensionless number
Ar = (d_s**3 * rho_f * (rho_s-rho_f) *g) / mu_f**2  # Archimedes number to characterize particle size
d_p_dimensionless = Ar ** (1/3)                     # Dimensionless solids
ut_dimensionless = ( 18/(d_p_dimensionless)**2 + (2.335-1.744*phi)/(d_p_dimensionless)**0.5 )**(-1)   # Dimensionless terminal velocity, Kunii P80, phi in range from 0.5 to 1
      
# Motion      
uf = 3.5          # Gas velocity (m/s)
ut = ut_dimensionless / ( rho_f**2/(mu_f*(rho_s-rho_f)*g))**(1/3)  # Terminal velocity of solids (m/s)
#ut = 2.5

########## Preparation of data, list array or dict ##########
# Geometry
layer_heights = np.zeros(n_heights)           # 1D array of bed heights (m)
layer_area = np.zeros(len(regions))           # 1D array for cross sectional area (m2)
layer_V = np.zeros((n_heights,len(regions)))    # 2D array for volume of layers (m3)
layer_heights[0] = 0.5                        # Height of dense bed
layer_heights[1:] = ( bed_height-layer_heights[0]) / (n_heights-1) # Height of the rest of bed
layer_area[core] = 2 * 3.5             # Area of core region
layer_area[annulus] = 2 * 0.5             # Area of annulus region
for i in range(n_heights):
    layer_V[i,core] = layer_heights[i] * layer_area[core]    # Volume of core layers
    layer_V[i,annulus] = layer_heights[i] * layer_area[annulus]    # Volume of annulus layers


def ds_conc_dt(t,c_flat):
    c = c_flat.reshape((n_heights,len(regions),len(s_species)))
    # s_conc = np.zeros((n_heights,len(regions),len(s_species)))        # Concentration for each layer and species (kg/m3)
    s_conc = c.copy()
    
    # Mass flow rate and Concentration
    s_min = np.zeros((n_heights,len(regions),len(s_species)))     # Inlet mass flow rate for each layer and species (kg/s) shape: (height,region,species)
    s_mout = np.zeros((n_heights,len(regions),len(s_species)))    # Outlet mass flow rate for each layer and species (kg/s)
    s_mlat = np.zeros((n_heights,len(s_species)))                 # Lateral mass flow rate for each layer and species, only from core to anuulus (kg/s)
    
    
    # For dense bed region, layer 0
    # Assume that the cell in anuulus region of dense bed does not exist, so no definition for min, mlat, mout[0,annulus,:]
    
    s_mout[0,core,Fe] = (layer_area[core]+layer_area[annulus]) * rho_s * uf * 0.059 * (uf / ut - 1)**2 
    
    #For the layers above dense bed
    for i in range(1, n_heights):
        s_mout[i,core,Fe] = s_conc[i,core,Fe] * (uf - ut) * layer_area[core]
        s_mout[i,annulus,Fe] = s_conc[i,annulus,Fe] * (uf - ut) * layer_area[annulus]
        s_min[i,core,Fe] = s_mout[i-1,core,Fe]
        s_mlat[i,Fe] = K * s_conc[i,core,Fe] 
    # Special treatment for annulus, no min for highest layer
    for i in range(n_heights-2, 0, -1):
        s_min[i,annulus,Fe] = s_mout[i+1,annulus,Fe]
    
    s_min[0,core,Fe] = s_mout[n_heights-1, core, Fe]           # Inlet mass flow, only from core flow

    
    
    # Definition of the derivative of concentration
    dcdt = np.zeros_like(c)
    # Dense bed, i = 0
    dcdt[0,core,Fe] = (s_min[0,core,Fe]+s_mout[1,annulus,Fe]-s_mout[0,core,Fe]) / (layer_V[0,core]+layer_V[0,annulus])


    # Layers above dense bed, i > o
    for i in range(1, n_heights):
        # Core region
        dcdt[i,core,Fe] = (s_min[i,core,Fe] - s_mlat[i,Fe] - s_mout[i,core,Fe]) / layer_V[i,core]
        # Annulus region
        if i == n_heights - 1:
            dcdt[i,annulus,Fe] = (s_mlat[i,Fe] - s_mout[i,annulus,Fe]) / layer_V[i,annulus]
        else:
            dcdt[i,annulus,Fe] = (s_min[i,annulus,Fe] + s_mlat[i,Fe] - s_mout[i,annulus,Fe]) / layer_V[i,annulus]
        
    return dcdt.flatten()

t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 600)

# Give a initial solids concentration
s_conc0 = np.zeros((n_heights,len(regions),len(s_species)))
for i in range(0, n_heights):
    if i == 0:
        s_conc0[i, core,Fe] = 2800
        s_conc0[i,annulus,Fe]= 2800
    else:
        s_conc0[i, core,Fe] = 50 * (n_heights - i)
        s_conc0[i,annulus,Fe]= 50 * (n_heights - i)

s_conc_sol = solve_ivp(ds_conc_dt, t_span, s_conc0.flatten(), t_eval=t_eval,method='BDF')
# sol.y.shape = (n_total_variables, n_time_points), n_total_variables = n_heights * n_regions * n_species
# sol.y.T transpose a matrix -> (n_time_points, n_total_variables)
# -1 : Let NumPy automatically infer the length of the first dimension from the total number of variables divided by the product of the following dimensions
# -1: Flexible if the number of t_eval is changed
s_conc = s_conc_sol.y.T.reshape((-1, n_heights, len(regions), len(s_species)))   # shape (time, height, region, species)
time = s_conc_sol.t
fe_conc = s_conc[:, :, :, Fe]  # shape (n_times, n_heights, n_regions)

m = np.zeros((len(time),n_heights,len(regions),len(s_species))) # kg total mass in each layers
F_out = np.zeros((len(time),n_heights,len(regions),len(s_species))) # kg/s outflow rate for each layer
F_in = np.zeros((len(time),n_heights,len(regions),len(s_species))) # kg/s outflow rate for each layer
F_lat = np.zeros((len(time),n_heights,len(s_species)))
total_mass = np.zeros((len(time), len(s_species)))           # kg total mass for all layers

##########################
F_out[:,0,core,Fe] = (layer_area[core]+layer_area[annulus]) * rho_s * uf * 0.059 * (uf / ut - 1)**2

for t in range(len(time)):
    for i in range(1,n_heights):
        F_out[t,i,core,Fe] = s_conc[t,i,core,Fe] * (uf - ut) * layer_area[core]
        F_out[t,i,annulus,Fe] = s_conc[t,i,annulus,Fe] * (uf - ut) * layer_area[annulus]
        F_in[t,i,core,Fe] = F_out[t,i-1,core,Fe]
        F_lat[t,i,Fe] = K * s_conc[t,i,core,Fe] 
        
    for i in range(n_heights-2, 0, -1):
        F_in[t,i,annulus,Fe] = F_out[t,i+1,annulus,Fe]
        
    F_in[t,0,core,Fe] = F_out[t,n_heights-1, core, Fe]           # Inlet mass flow, only from core flow
    
    
for t in range(len(time)):
    m[t,0,core,Fe] = s_conc[t,0,core,Fe] * (layer_V[0,core] )
    m[t,0,annulus,Fe] = s_conc[t,0,core,Fe] * (layer_V[0,annulus])
    
    for i in range(1,n_heights):
        for j in range(len(regions)):
            m[t,i,j,Fe] = s_conc[t,i,j,Fe] * layer_V[i,j]
            
for t in range(len(time)):
    total_mass[t,Fe] = np.sum(m[t,1:,:,Fe])
    total_mass[t,Fe] += m[t,0,core,Fe]
    total_mass[t,Fe] += m[t,0,annulus,Fe]



                
                

Fe_F_in = F_in[:,:,:,0]
Fe_F_out = F_out[:,:,:,0]
Fe_m = m[:,:,:,0]


for k, sp in enumerate(s_species):
    plt.plot(time, total_mass[:, k], label=sp)

plt.xlabel('Time (s)')
plt.ylabel('Total Mass (kg)')
plt.title('Total Mass of Each Species Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



for j, region in enumerate(regions):        # j = 0: core, 1: annulus
    for i in range(n_heights):                   # i = height layer index
        plt.figure(figsize=(8, 5))
        if j == core:
            plt.plot(time, F_in[:, i, j, Fe],'-.', label='F_in')
            plt.plot(time, (F_out[:, i, j, Fe] + F_lat[:,i,Fe]), '--', label='F_out')
        if j == annulus:
            plt.plot(time, (F_in[:, i, j, Fe] +  + F_lat[:,i,Fe] ),'-.', label='F_in')
            plt.plot(time, F_out[:, i, j, Fe], '--', label='F_out')
        
        plt.plot(time, m[:, i, j, Fe], label='Mass')

        plt.xlabel('Time (s)')
        plt.ylabel('Fe (kg or kg/s)')
        plt.title(f'Fe in {region.capitalize()} - Layer {i}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



for region_idx, region in enumerate(regions):
    plt.figure()
    for h in range(n_heights):
        plt.plot(time,
                 fe_conc[:, h, region_idx],
                 label=f'Layer {h} ({region})')
    plt.xlabel('Time (s)')
    plt.ylabel('Fe Concentration (kg/m³)')
    plt.title(f'Fe in {region.capitalize()} Region')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

dt = np.diff(time)  # time step array, shape = (n_times - 1,)

mass_balance_error = np.zeros((len(time), n_heights, len(regions)))  # Optional: only check for Fe

# Check accumulation for Fe only
for i in range(1, len(time)):
    for h in range(n_heights):
        for r in range(len(regions)):
            # mass change (kg)
            dm = m[i, h, r, Fe] - m[i-1, h, r, Fe]
            
            # net inflow = (F_in - F_out) * dt (kg)
            if r == core:
                net_flux = (F_in[i-1, h, r, Fe] - (F_out[i-1, h, r, Fe] + F_lat[i-1,h,Fe])) * dt[i-1]
            if r == annulus:
                net_flux = ((F_in[i-1, h, r, Fe] + F_lat[i-1,h,Fe])- F_out[i-1, h, r, Fe]) * dt[i-1]
            
            
            
            
            # compare
            mass_balance_error[i, h, r] = dm - net_flux 

# Optional: plot or print the max error
print("Maximum mass balance error (kg):", np.abs(mass_balance_error).max())
plt.figure()
for h in range(n_heights):
    for r in range(len(regions)):
        label = f'Layer {h} {regions[r].capitalize()}'
        plt.plot(time[1:], np.abs(mass_balance_error[1:, h, r]), label=label)

plt.xlabel("Time (s)")
plt.ylabel("Mass balance error (kg)")
plt.title("Mass balance error over time for Fe")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


    
