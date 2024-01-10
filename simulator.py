

# Imports
import os

import inspect

from datetime import date

import pandas as pd

import numpy as np

from math import ceil

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = ["Latin Modern Roman"]

# Define large default argument variables
default_debug_ctl = { # Default debug control for class
    'sim': False, # Debug prints in the class
    'file': False # Save debug prints to text file
}

default_output_ctl = { # Default simulation output control for class
    'filepath': 'default', # Path for output file, codes unset
    'filename': 'default', # Name for output file, codes unset
    'file': True, # Save output file
    'plots': False # Save state plot
}

# Class Definition
class Simulator:
    '''
    Class used to perform numerical integration to solve a system of ODEs
    forward in time. Typically representing the states of some vehicle or 
    dynamic system.

    Notes:

    '''

    # Class-wide variables
    default_results_folder = 'Default Simulation Results'

    def __init__(self, tspan, x0, algorithm='rk4', total_steps=-1, 
        timestep=-0.1, state_names='default', output_ctl=default_output_ctl, 
        debug_ctl=default_debug_ctl):
        '''
        Constructor method

        Required Inputs:
            tspan: tuple of length 2, (time_start, time_stop) in seconds
            x0: 1d numpy array of initial state values
            sysdyn: function call accepting states and t, returning dx_dt

        Optional Inputs:
            algorithm: str with numerical integration algorithm name,
                default = 'rk4', options: ('rk4')
            total_steps: int number of steps in integration,
                default = -1 indicates unset (set to 100)
            timestep: float of timestep in seconds, default = -0.1
                indicates unset, set by total_steps or default
            state_names: list of state names for the system, will become the 
                column names in the state trajectory dataframe. Default set
                to 'default' string indicates unset, and incremental 'x_' will
                be used for numbering
            output _ctl: dict with fields controlling file and plot outputs
                of the simulation
            debug_ctl: dict with boolean fields controlling debug printing,
                default 'sim' and 'file' both false
            state_mod_fun: function call accepting index i for direct change
                of system states in simulation
        '''

        # Assign instance variables
        self.tspan = tspan
        self.tstart = tspan[0]
        self.tstop = tspan[1]

        self.algorithm = algorithm

        # self.state_mod_fun = state_mod_fun

        self.debug_ctl = debug_ctl
        self.debug_statements = [] # Init empty list to track debug prints

        # Compute timing details for simulation instance
        if (total_steps!=-1) & (timestep!=-0.1): # Handle over-defined case

            raise Exception('RK4 input error, timing over-defined')

        elif total_steps != -1: # Defined by total steps
            
            self.n = total_steps # Set to system number of steps  

        elif timestep != -0.1: # Defined by timestep size

            self.n = ceil((self.tstop - self.tstart) / timestep) 

        else: # Default undefined case

            self.n = 100 # Set to system number of steps

        # Set time vector and timestep
        self.t_vec, self.dt = np.linspace(self.tstart, self.tstop, self.n, 
                                     retstep=True)

        if self.debug_ctl['sim']:
            msg = ("Sim timing: n = %0.1f, dt = %0.3f") % (self.n, self.dt)
            self.debug_statements.append(msg)
            print(msg)

        # Initialize state-trajectory tracking matrix
        if state_names == 'default':
            col_names = [('x' + str(i)) for i in range(len(x0))]
        else:
            col_names=state_names
            
        col_names.insert(0, 't')
        self.col_names = col_names

        # Create state trajectory dataframe to fill
        self.state_traj = pd.DataFrame(columns=self.col_names, 
                                       index=range(self.n))

        # Fill in time vector and initial states
        self.state_traj['t'] = self.t_vec
        self.state_traj.iloc[0, 1:] = x0

        # Set output information for file and plots

        # Plot information omitted for now (5/12/23)

        self.output_file_bool = output_ctl['file'] # T/F on saving output file

        if self.output_file_bool:
            self.set_output_file(output_ctl)

        self.term_time = 0 # Initialize timestamp for term check to 0

    def compute(self, *args, desc='Sim Outer Loop'):
        '''
        Method to perform numerical integration simulation
        '''

        for i in tqdm(range(self.n - 1), desc):

            self.rk4_step(i, args)

            # Add optional direct state-change function
            t = self.state_traj.iloc[i+1, 0]
            x = self.state_traj.iloc[[i+1], 1:].to_numpy().flatten()
            x_mod = self.state_mod_fun(t, x, i)
            self.state_traj.iloc[[i+1], 1:] = x_mod

            # Check early termination function
            x_term = self.state_traj.iloc[[i+1], 1:].to_numpy().flatten()
            term_check = self.term_check(t, x_term)
            if term_check:

                if i < (self.n - 1): # If before the last step trim output
                    self.state_traj = self.state_traj.iloc[:i+2]

                break

        if self.debug_ctl['file']: # If debug save to file

            output_fullpath = os.path.join(self.output_filepath, 
                                                'debug_print.txt') 

            with open(output_fullpath, 'w') as fp:

                for item in self.debug_statements:

                    # write each item on a new line
                    fp.write("%s\n" % item)

        if self.output_file_bool: # If output file desired

            output_fullpath = os.path.join(self.output_filepath, 
                                                self.output_filename) 

            self.state_traj.to_csv(output_fullpath, index=None)

    def rk4_step(self, i, *args):
        '''
        Single timestep Runge-Kutta 4th/5th order numerical integration method

        RK4 Reference:
        S. C. Chapra and R. Canale, “Numerical Methods for Engineers,” 
        5th Edition, McGraw-Hill (International Edition), New York, 2006.
        - Chapter 25.3.3
        '''

        x = self.state_traj.iloc[[i], 1:].to_numpy().flatten() # x this step
        t = self.state_traj.iloc[i, 0] # Time this step
        dt = self.dt

        args = args[0] # Unwrap one tuple layer

        if len(args) > 0: # If some extra input arguments

            # Perform RK4 computation
            k1 = dt*self.sysdyn(t, x, args)
            k2 = dt*self.sysdyn(t + dt/2.0, x + k1/2.0, args)
            k3 = dt*self.sysdyn(t + dt/2.0, x + k2/2.0, args)
            k4 = dt*self.sysdyn(t + dt, x + k3, args)

        else: # If no extra input arguments

            # Perform RK4 computation
            k1 = dt*self.sysdyn(t, x)
            k2 = dt*self.sysdyn(t + dt/2.0, x + k1/2.0)
            k3 = dt*self.sysdyn(t + dt/2.0, x + k2/2.0)
            k4 = dt*self.sysdyn(t + dt, x + k3)

        x_step = x + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0

        # print(x_step)
        
        self.state_traj.iloc[i + 1, 1:] = x_step # Assign to next row

    def sysdyn(self, t, x):
        '''
        Default system dynamics method for the simulation, x_dot equals 0
        '''

        dx_dt = np.zeros(x.size)

        return dx_dt

    def state_mod_fun(self, t, state_vec, i):
        '''
        Default state modification method for the simulation, takes no action 
        '''

        return state_vec
    
    def term_check(self, t, state_vec, *args, **kwargs):
        '''
        Default termination check function, returns false
        '''

        return False
    
    def load_x0(self, x0: np.ndarray) -> None:
        '''
        Method to reset the simulation and load in a new x0 vector
        '''

        # Create state trajectory dataframe to fill
        self.state_traj = pd.DataFrame(columns=self.col_names, 
                                       index=range(self.n))

        # Fill in time vector and initial states
        self.state_traj['t'] = self.t_vec
        self.state_traj.iloc[0, 1:] = x0

    def load_tspan(self, tspan: tuple, total_steps: int = -1, 
                   timestep: float = -0.1) -> None:
        '''
        Method to take in a new timespand for the simulation, this will reset
        the time and trajectory dataframes with the new information
        '''

        # Assign instance variables
        self.tspan = tspan
        self.tstart = tspan[0]
        self.tstop = tspan[1]

        # Compute timing details for simulation instance
        if (total_steps!=-1) & (timestep!=-0.1): # Handle over-defined case

            raise Exception('RK4 input error, timing over-defined')

        elif total_steps != -1: # Defined by total steps
            
            self.n = total_steps # Set to system number of steps  

        elif timestep != -0.1: # Defined by timestep size

            self.n = ceil((self.tstop - self.tstart) / timestep) 

        else: # Default undefined case

            self.n = 100 # Set to system number of steps

        # Set time vector and timestep
        self.t_vec, self.dt = np.linspace(self.tstart, self.tstop, self.n, 
                                          retstep=True)
        
        # Save existing x0
        x0_old = self.state_traj.iloc[0, 1:]

        # Create state trajectory dataframe to fill with new timespan
        self.state_traj = pd.DataFrame(columns=self.col_names, 
                                       index=range(self.n))

        # Fill in time vector and replace initial states
        self.state_traj['t'] = self.t_vec
        self.state_traj.iloc[0, 1:] = x0_old

    def set_output_file(self, output_ctl):
        '''
        Method to determine the path and filename default for sim results files
        '''

        # Set output file path
        if output_ctl['filepath'] == 'default': # Default path location

            # Find pathname and directory of calling file
            fullpath_calling = os.path.abspath((inspect.stack()[2])[1])
            dir_calling = os.path.dirname(fullpath_calling)

            # Get contents of calling directory
            contents_calling = os.listdir(dir_calling)

            # Create output file path to default folder
            self.output_filepath = os.path.join(dir_calling, 
                                                self.default_results_folder) 

            # Create default results directory if it does not exist
            if self.default_results_folder not in contents_calling:

                os.mkdir(self.output_filepath)

        else: # User-input path location
            self.output_filepath = output_ctl['filepath']

        # Set output filename
        if output_ctl['filename'] == 'default': # Default file name
            # Default file name: 'YYYY_MM_DD_#' where '#' is incremented above
            # other files of the same date in the directory

            # Get contents of output file path directory
            result_dir_contents = os.listdir(self.output_filepath)

            # Form date component of filename string
            today = date.today()
            date_str = today.strftime("%Y_%m_%d")

            # Determine index number for this file
            if any(date_str in cont for cont in result_dir_contents):

                # Select the filenames that match this date
                matching_names = [cont for cont in result_dir_contents if 
                                  date_str in cont]

                # Isolate the index numbers for these filenames
                index_str = [sub.replace(date_str + '_', '') for 
                           sub in matching_names]
                
                indexes = [eval(i) for i in index_str] # Convert to int
                res_idx = max(indexes) + 1 # Increment by 1

            else: # This is the first result file

                res_idx = 0 # Start at 0

            # Set default filename for this file
            self.output_filename = date_str + '_' + str(res_idx) + '.csv'

        else: # User-input file name
            self.output_filename = output_ctl['filename']

    def plot_states(self):
        '''
        Method to perform basic plotting of all simulation states, using 
        pandas built-in plotting method
        '''

        state_plot = self.state_traj.plot(x='t' ,grid=True, legend=True, 
            title='State Trajectories', xlabel='Time (s)', 
            ylabel='State Values')
        plt.show()

# ------------------ Useful Numerical Integration Functions ------------------ #
def rk4_step_fun(sysdyn, x, y, dy, *args):
    '''
    Single timestep Runge-Kutta 4th/5th order numerical integration method,
    setup as a function for external use
    '''

    # Perform RK4 computation

    if len(args) > 0: # If some extra input arguments

        k1 = dy*sysdyn(y, x, args)
        k2 = dy*sysdyn(y + dy/2.0, x + k1/2.0, args)
        k3 = dy*sysdyn(y + dy/2.0, x + k2/2.0, args)
        k4 = dy*sysdyn(y + dy, x + k3, args)

    else: # If no extra input arguments

        k1 = dy*sysdyn(y, x)
        k2 = dy*sysdyn(y + dy/2.0, x + k1/2.0)
        k3 = dy*sysdyn(y + dy/2.0, x + k2/2.0)
        k4 = dy*sysdyn(y + dy, x + k3)

    x_step = x + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0
    
    return x_step





















