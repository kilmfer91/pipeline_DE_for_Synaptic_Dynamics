# import libraries
from utils import *
from libraries.params_fitting import Fit_params_SD
from synaptic_dynamic_models.MSSM import MSSM_model
from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.LAP import LAP_model
from synaptic_dynamic_models.SynDynModel import SynDynModel

# **********************************************************************************************************************
# PATHS TO STORE/LOAD DATA
path_reference_data = "../reference_data/"
path_store_models = "../outputs/fitting_adjusted_MSSM/"

# **********************************************************************************************************************
# GLOBAL PARAMETERS
model_str = "MSSM"  # String defining the model to use (e.g. MSSM, TM, LAP)

# AUXILIAR VARIABLES FOR THE EXAMPLE
ind = 0                 # Auxiliar index to define the Synaptic Dynamics mechanism (0: depression, 1: facilitation)
ind_experiment = 3      # Index of experiment to run/load
num_experiments = 1     # Number of experiments to run
prefix = prefix_v[ind]  # string defining a description of the SD model: "depression" or "facilitation"

# ARGUMENTS FOR THE SIMULATION SETUP: list following this order [sfreq, max_t, input_factor, output_factor, description]
# sfreq (int) sampling frequency
# max_t (float) simulation time [in seconds]
# input_factor (float) scaling factor of input signal
# output_factor (float) scaling factor of reference signal
# description (String) sufix for the name of the file, that is stored once the fitting process finishes.
sim_args = [sfreq_ext[ind], max_t_ext[ind], input_factor_ext[ind], output_factor_ext[ind], str(ind)]

# ******************************************************************************************************************
# DICTIONARY OF PARAMETERS
example_fitting = Example_fitting(model_str, path_signals=path_reference_data, path_outputs=path_store_models)
example_fitting.initial_params(ind, sim_args)
example_fitting.params_DE()
dict_params = example_fitting.dict_params
sim_params = example_fitting.sim_params
DE_params = ['best1bin', 1000, 15, 0.01, (0.5, 1), 0.7, None, None, True, False, 'latinhypercube', 0, 'deferred',
             1, (), None, None, True]

# ******************************************************************************************************************
# LOADING INPUT AND REFERENCE SIGNALS
input_signal = loadObject(prefix + "_input_lf", path_reference_data)
reference_signal = loadObject(prefix + "_epsc_lf", path_reference_data)

# **********************************************************************************************************************
# SYNAPTIC DYNAMICS MODEL
model_SD = None
num_syn = int(len(dict_params['params_name']) * DE_params[2])
if model_str == "MSSM":
    model_SD = MSSM_model(n_syn=num_syn)
if model_str == "TM":
    model_SD = TM_model(n_syn=num_syn)
if model_str == "LAP":
    model_SD = LAP_model(n_syn=num_syn)
# Setting simulation parameters
model_SD.set_simulation_params(sim_params)

# **********************************************************************************************************************
# FITTING PROCESS

# Setting vars for multiple fitting experiments
experiment_loop_lims = [ind_experiment, ind_experiment + num_experiments]

# Initialization
pf = Fit_params_SD(np.roll(input_signal, 1), reference_signal, model_SD, dict_params, DE_params)

# Time measurements
ini_loop_time = m_time()
# Loop through experiments to perform
stop_flag = False
for exp_ind_experiment in range(experiment_loop_lims[0], experiment_loop_lims[1]):
    print("Evaluating curve fitting for ", prefix, ", index ", str(exp_ind_experiment))

    # Updating index to save
    dict_params['ind_experiment'] = exp_ind_experiment

    # If num_experiments are reached, stop fitting process
    if stop_flag:
        break

    # Fitting parameters
    tuned_parameters = pf.run_fit()

    # Evaluating stop condition
    if exp_ind_experiment > experiment_loop_lims[1] - 1:
        stop_flag = True

# Evaluating Model
output_model = pf.evaluate_model(tuned_parameters.T)

# **********************************************************************************************************************
# Time measurements
end_one_loop_time = m_time()
msg_time = "Loading stored params"
if pf.fit_new_params:
    msg_time = "Fitting new params"
print_time(end_one_loop_time - ini_loop_time, msg_time)

# Printing fitted parameters of the SD model
print("Parameters found by the pipeline for the " + model_str + " model:\n",
      example_fitting.dict_params['params_name'], "\n", tuned_parameters)

# Plot of the postsynaptic response
label = prefix + r" - $E_{PSC}(t)$ for input at " + str(r_ext[ind]) + "Hz"
figsize = (10, 7.2)  # (10, 4.8)  # (4.8, 7.2)
fig = plt.figure(figsize=figsize)
plt.suptitle(label, fontsize=12)
plot_temp_mssm(pf.model_SD, fig, ind_interest=None)
