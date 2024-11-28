import matplotlib.pyplot as plt
import numpy as np

# import libraries
from libraries.params_fitting import Fit_params_stp
from synaptic_dynamic_models.MSSM import MSSM_model
from synaptic_dynamic_models.TM import TM_model
from utils import *

# **********************************************************************************************************************
# GLOBAL PARAMETERS
model_str = "TM"  # String defining the model to use (e.g. MSSM, TM)
num_syn = 1       # Number of synapses to simulate
ind = 0           # Auxiliar index to define the Synaptic Dynamics mechanism (0: depression, 1: facilitation)

# VARIABLES TO PROCESS
prefix = prefix_v[ind]
ext_params = False
exp_ind_save = 0

# PATHS TO STORE/LOAD DATA
path_reference_data = "reference_data/"
path_store_models = "outputs/fitting_test/"

# ******************************************************************************************************************
# DICTIONARY OF PARAMETERS
example_fitting = Example_fitting(model_str, path_signals=path_reference_data, path_outputs=path_store_models)
example_fitting.initial_params(ind, description=str(ind))  # TRANSFORM INTO *ARGS FUNCTION WITH r,sfreq,max_t,end_t,input_factor,output_factor
dict_params = example_fitting.dict_params
sim_params = example_fitting.sim_params

# ******************************************************************************************************************
# Loading reference signals
input_signal = loadObject(prefix + "_input_lf", path_reference_data)
reference_signal = loadObject(prefix + "_epsc_lf", path_reference_data)

# **********************************************************************************************************************
# FITTING PROCESS
# Time measurements
ini_loop_time = m_time()
if ext_params:
    # ******************************************************************************************************************
    # Creating model with external parameters
    model_stp = None
    """
    if model_str == "MSSM":
        model_stp = MSSM_model(n_syn=num_syn)
    if model_str == "LAP":
        model_stp = LAP_model(n_syn=num_syn)
    if model_str == "TM":
        model_stp = TM_model(n_syn=num_syn)

    model_stp.set_simulation_params(sim_params)

    if stochastic_analysis:
        for i in range(num_trials_stochastic):
            seed = int(su([1, 10e4]))
            print("freq analaysis No. ", i, "seed ", seed)
            Input = poisson_generator(dt, time_vector, rate=r, n=1, myseed=seed)[0, :]

            kwargs_model = {'model': model_stp, 'Input': Input, 'params_name': ext_pa_name,
                            'mode': dict_params['ODE_mode'], 'only spikes': dict_params['only_spikes']}
            # Evaluating the model
            if model_str == "MSSM":
                model_mssm(time_vector, *ext_pa, **kwargs_model)
            if model_str == "LAP":
                model_lap(time_vector, *ext_pa, **kwargs_model)
            if model_str == "TM":
                model_tm(time_vector, *ext_pa, **kwargs_model)

            trials_spike = trials_spike + model_stp.output_spike_events
            trials_time_spike = trials_time_spike + model_stp.time_spike_events
            trials_first_spike.append(model_stp.output_spike_events[0])
        trials_spike = np.array(trials_spike)
        trials_time_spike = np.array(trials_time_spike)
        trials_first_spike = np.array(trials_first_spike)
    else:
        # Deterministic input, executing only once
        kwargs_model = {'model': model_stp, 'Input': Input, 'params_name': ext_pa_name,
                        'mode': dict_params['ODE_mode'], 'only spikes': dict_params['only_spikes']}
        # Evaluating the model
        if model_str == "MSSM":
            model_mssm(time_vector, *ext_pa, **kwargs_model)
        if model_str == "LAP":
            model_lap(time_vector, *ext_pa, **kwargs_model)
        if model_str == "TM":
            model_tm(time_vector, *ext_pa, **kwargs_model)

    # print(rmse(model_stp.get_output(), -epsc_ref))
    # Time measurements
    end_one_loop_time = m_time()
    print_time(end_one_loop_time - ini_loop_time, "Loading external/manual params")
    # """
else:
    # ******************************************************************************************************************
    # Fitting parameters
    pf = Fit_params_stp(sim_params, dict_params)
    pf.set_input(input_signal)

    print("Evaluating curve fitting for ", prefix, ", index ", str(exp_ind_save))

    # Updating index to save
    dict_params['exp_ind_save'] = exp_ind_save

    # Fitting parameters
    pf.run_fit(reference_signal, dict_params)

    # Evaluating Model
    output_model = pf.evaluate_model()

    # Computing efficacy and RMSE value
    _ = pf.model_stp.get_efficacy_time_ss(num_syn, pf.model_stp.output_spike_events)

    model_stp = pf.model_stp
    model_str = pf.model_str

    # Time measurements
    end_one_loop_time = m_time()
    msg_time = "Loading stored params"
    if pf.fit_new_params:
        msg_time = "Fitting new params"
    print_time(end_one_loop_time - ini_loop_time, msg_time)


label = r"Facilitation - $E_{PSC}(t)$ for input at " + str(r_ext[ind]) + "Hz"
plot_res_mssm(model_stp.time_vector, input_signal, reference_signal, label)