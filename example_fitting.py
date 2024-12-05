# import libraries
from utils import *
from libraries.params_fitting import Fit_params_SD
from synaptic_dynamic_models.MSSM import MSSM_model
from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.SynDynModel import SynDynModel

# **********************************************************************************************************************
# PATHS TO STORE/LOAD DATA
path_reference_data = "reference_data/"
path_store_models = "outputs/fitting_test/"

# **********************************************************************************************************************
# GLOBAL PARAMETERS
model_str = "MSSM"  # String defining the model to use (e.g. MSSM, TM)

# AUXILIAR VARIABLES FOR THE EXAMPLE
ind = 0                 # Auxiliar index to define the Synaptic Dynamics mechanism (0: depression, 1: facilitation)
ind_experiment = 0      # Index of experiment to run/load
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
DE_params = example_fitting.DE_params

# ******************************************************************************************************************
# LOADING INPUT AND REFERENCE SIGNALS
input_signal = loadObject(prefix + "_input_lf", path_reference_data)
reference_signal = loadObject(prefix + "_epsc_lf", path_reference_data)

# **********************************************************************************************************************
# SYNAPTIC DYNAMICS MODEL
model_SD = None
if model_str == "MSSM":
    model_SD = MSSM_model()
if model_str == "TM":
    model_SD = TM_model()
# Setting simulation parameters
model_SD.set_simulation_params(sim_params)

# **********************************************************************************************************************
# FITTING PROCESS
# Time measurements
ini_loop_time = m_time()
print("Evaluating curve fitting for", prefix, ", index ", str(ind_experiment))

# Initialization
pf = Fit_params_SD(input_signal, reference_signal, model_SD, dict_params, DE_params)

# In case of running multiple experiments, update the key 'ind_experiment' of dict_params and call set_dict_params()
dict_params['ind_experiment'] = ind_experiment
pf.set_dict_params(dict_params)

# Fitting parameters
tuned_parameters = pf.run_fit()

# Evaluating Model
output_model = pf.evaluate_model()

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
plot_res_mssm(pf.model_SD.time_vector, input_signal, np.squeeze(pf.model_SD.get_output()), label)
