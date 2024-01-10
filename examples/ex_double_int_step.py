
# Example of the numerical integrator, double integrator dynamics step response

# Standard imports
import numpy as np

# Add package directory to the path
import os
import sys
sim_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, sim_pkg_path)

# Workspace package imports
import simulator

# Example system dynamics function
def sysdyn_spring_mass(t, x, *args):

    # Set system constants
    m = 2
    k = 1.5
    d = 0.5
    r = np.array([1, 0])

    # Set system gains
    k1 = 10
    k2 = 2.5
    ki = 2
    K = np.array([[k1], [k2]])

    # Form system matrices
    A = np.array([[0, 1], [-k/m, -d/m]])
    B = np.array([0, 1])

    # Compute the controller
    e = r - x
    sysdyn_spring_mass.e_int += (t - sysdyn_spring_mass.t_last)*e[0]
    sysdyn_spring_mass.t_last = t
    u = K.T@e + ki*sysdyn_spring_mass.e_int

    # Compute the system dynamics
    dx = A@x + B*u

    return dx

sysdyn_spring_mass.e_int = 0
sysdyn_spring_mass.t_last = 0

# Setup simulation output parameters to test all options
output_ctl1 = { # Default filename custom filepath
    'filepath': '<absolute path>', # Path for output file, codes unset
    'filename': 'default', # Name for output file, codes unset
    'file': True, # Save output file
    'plots': True # Save output plots
}

output_ctl2 = { # Custom filename default filepath
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'name_test', # Name for output file, codes unset
    'file': True, # Save output file
    'plots': True # Save output plots
}

output_ctl3 = { # Custom filename custom filepath
    'filepath': '<absolute path>', # Path for output file, codes unset
    'filename': 'name_test', # Name for output file, codes unset
    'file': True, # Save output file
    'plots': True # Save output plots
}

# Setup simulation debug parameters to test all options
debug_ctl1 = { # Print debug statements
    'sim': True, # Debug prints in the class
    'file': False # Save debug prints to text file
}

debug_ctl2 = { # Print and save debug statements
    'sim': True, # Debug prints in the class
    'file': True # Save debug prints to text file
}

# Setup simulation parameters to test all timing options
tspan = (0, 10)
x0 = np.array([0, 0])
timestep = 0.01
number_steps = 250
state_names = ['position', 'velocity']

# Run the simulation and plot state trajectories
sim_test = simulator.Simulator(tspan, x0, sysdyn_spring_mass, timestep=timestep,
                               state_names=state_names)
sim_test.compute()
sim_test.plot_states()


















