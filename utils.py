import sys
import os
import time
import re
from datetime import timedelta
import pickle

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# **********************************************************************************************************************
# PATHS TO STORE/LOAD DATA
path_signals_ext = "../../reference data/"
path_outputs_ext = "../../outputs/fitting_test/"

# **********************************************************************************************************************
# EXAMPLE OF PARAMETERS FOR MSSM AND TM MODELS
"""MSSM"""
# External parameters for MSSM with 1 synapse. Facilitation
params_name_mssm = ['tau_c', 'alpha', 'V0', 'tau_v', 'P0', 'k_NtV', 'k_Nt', 'tau_Nt', 'k_EPSP', 'tao_EPSP']
ext_par_mssm = [3e-3, 905e-4, 3.45, 8.4e-3, 0.002, 1, 1, 13.5e-3, 10, 9e-3, 0.0, 0.0]  # From Karim C0 0.05

"""TM Model"""
# From the paper "Differential signaling via the same axon of neocortical pyramidal neurons" Markram, Wang, Tsodyks
params_name_tm = ['U0', 'tau_in', 'tau_rec', 'Ase', 'tau_syn']
ext_par_tm = [0.03, 530e-3, 130e-3, 1540, 2.5e-3]

# labels for uploading examples of facilitation and depression
prefix_v = ['depression', 'facilitation']

# ******************************************************************************************************************
# TIME CONDITIONS
r_ext = [10, 30]  # Frequency of the spike train (Hz)
sfreq_ext = [2e3, 5e3]  # Sampling frequency (Hz)
ini_t_ext = [.1, .04]  # Initial time of simulation (s)
end_t_ext = [3.1, .55]  # Maximum time of spike train (s) (MUST BE LESS THAN max_t_v)
max_t_ext = [3.5, .8]  # Maximum time of simulation (s)
# Input-Output factors
input_factor_ext = [1.0, 1.0]
output_factor_ext = [1.0, 1.0]


# **********************************************************************************************************************
# MODELS OF SHORT-TERM PLASTICITY
def model_lap(time_vector, *args, **kwargs):
    """
    Simulation of a LAP synapses for a given input (kwargs['Input']), a given set of parameters
    (names of parameters - kwargs['params_name'], parameters - args), and the way to compute the time functions
    (kwargs['ODE_model']). An internal computation is done to represent the parameters as a dictionary and feed the LAP
    model, for example:
    param = {'Cai_0': args[0], 'KCa': args[1], 'tao_Cai': args[2], 'Krel_half': args[3], 'Krecov_0': args[4],
             'Krecov_max': args[5], 'Prel_0': args[6], 'Prel_max': args[7], 'Krecov_half': args[8], 'Krel': args[9],
             'Krel2': args[10], 'tao_EPSC': args[11], 'KGlu': args[12], 'n': args[13], 'Ntotal': args[14],
             'Cai0': args[15]}
    :param time_vector: (numpy array [t, ]) time vector for simulation.
    :param args: (parameters) set of parameters for the LAP model.
    :param kwargs: (dictionary {'model': a,  'Input': b, 'params_name': c, 'mode': d, 'only spikes': e})
                   Extra parameters to specify:
                   a (LAP_model) instance of LAP_model class.
                   b (numpy array [t, ]) the input spike train.
                   c (list) the name of parameters. It must have the same size as *args, and it must  contain at least
                   one of the following names:
                     ['Cai_0', 'KCa', 'tao_Cai', 'Krel_half', 'Krecov_0', 'Krecov_max', 'Prel_0', 'Prel_max',
                      'Krecov_half', 'Krel', 'Krel2', 'tao_EPSC', 'KGlu', 'n', 'Ntotal', 'Cai0']
                   d (String) the type of ODE solver supported by MSSM class. Up to now, there are two options:
                     'ODE': for Euler method.
                     'Analytical': for analytical solution of equations (no ODE solver).
                   e (boolean) if True, the output of the model is only the spike response of EPSP, otherwise the output
                     is the timeseries for EPSP.
    :return: mssm.N (numpy array [t, ]) Output of neurotransmitters buffering.
    """
    # Local variables
    params_name = kwargs['params_name']
    Input = kwargs['Input']
    output_model = None
    lap = kwargs['model']
    param = {}
    return_only_spike_event = False
    output_spike_events = []
    L = len(time_vector)

    # Creating param dictionary
    for i in range(len(args)):
        if params_name[i] == 'Prel_0':
            params_name[i] = 'Prel0'
        param[params_name[i]] = args[i]

    # Update parameters and initial conditions
    lap.set_model_params(param)
    lap.set_initial_conditions()

    # If there is a neuron model
    model_neuron = None
    if "model_neuron" in kwargs:
        if kwargs['model_neuron'] is not None:
            model_neuron = kwargs['model_neuron']
            model_neuron.set_simulation_params()

    # Returning the entire time series or only the response after an input spike
    if "only spikes" in kwargs:
        # print("Utils.py, model_lap(), only spikes is key in kwargs")
        return_only_spike_event = kwargs['only spikes']
    # Evaluating LAP model
    if kwargs['mode'] == "odeint":
        lap.evaluate_odeint_time(time_vector, Input)
    else:
        for t in range(L):
            if kwargs['mode'] == "Analytical":
                lap.evaluate_model_analytical(Input[t], t)
            elif kwargs['mode'] == "ODE":
                lap.evaluate_model_euler(Input[t], t)
            # elif kwargs['mode'] == "odeint":
            #     lap.evaluate_odeint(t, [time_vector[t - 1], time_vector[t]], Input[t])
            else:
                assert False, "'ODE mode' must be either 'Analytical' or 'ODE'"
            # Evaluating neuron response
            if model_neuron is not None:
                model_neuron.update_state(lap.EPSC[:, t], t)
                # Output based on the neuron model
                output_model = model_neuron.membrane_potential
            else:
                # Output based on the TM model
                output_model = lap.EPSC

            # Detecting spike events and storing TM output
            lap.detect_spike_event(t, output_model)

        # Computing output spike event in the last ISI
        t = L
        spike_range = (lap.time_spike_events[-1], t)
        lap.compute_output_spike_event(spike_range, output_model)

    if return_only_spike_event:
        return np.array(lap.output_spike_events)
    if model_neuron is not None:
        return model_neuron.membrane_potential
    else:
        return lap.EPSC


def model_tm(time_vector, *args, **kwargs):
    params_name = kwargs['params_name']
    Input = kwargs['Input']
    output_model = None
    tm = kwargs['model']
    param = {}
    return_only_spike_event = False
    output_spike_events = []
    L = len(time_vector)
    # Creating param dictionary
    for i in range(len(args)):
        param[params_name[i]] = args[i]

    # Update parameters and initial conditions
    tm.set_model_params(param)
    tm.set_initial_conditions()

    # If there is a neuron model
    model_neuron = None
    if "model_neuron" in kwargs:
        if kwargs['model_neuron'] is not None:
            model_neuron = kwargs['model_neuron']
            model_neuron.set_simulation_params()

    # Returning the entire time series or only the response after an input spike
    if "only spikes" in kwargs:
        # print("Utils.py, model_mssm(), only spikes is key in kwargs")
        return_only_spike_event = kwargs['only spikes']

    if kwargs['mode'] == "odeint":
        pass
        # tm.evaluate_odeint_time(time_vector, Input)
    else:
        for t in range(L):
            if kwargs['mode'] == "Analytical":
                tm.evaluate_model_analytical(Input[t], t)
            elif kwargs['mode'] == "ODE":
                tm.evaluate_model_euler(Input[t], t)

            # elif kwargs['mode'] == "odeint":
            #     tm.evaluate_odeint(t, [time_vector[t - 1], time_vector[t]], Input[t])
            else:
                assert False, "'ODE mode' must be either 'Analytical' or 'ODE'"

            # Evaluating neuron response
            if model_neuron is not None:
                model_neuron.update_state(tm.I_out2[:, t], t)
                # Output based on the neuron model
                output_model = model_neuron.membrane_potential
            else:
                # Output based on the TM model
                output_model = tm.I_out2

            # Detecting spike events and storing TM output
            tm.detect_spike_event(t, output_model)

        # Computing output spike event in the last ISI
        t = L
        spike_range = (tm.time_spike_events[-1], t)
        tm.compute_output_spike_event(spike_range, output_model)

    if return_only_spike_event:
        # return np.array(output_spike_events)
        return np.array(tm.output_spike_events)
    if model_neuron is not None:
        return model_neuron.membrane_potential
    else:
        return tm.I_out2


def model_mssm(time_vector, *args, **kwargs):
    """
    Simulation of a MSSM synapses for a given input (kwargs['Input']), a given set of parameters
    (names of parameters - kwargs['params_name'], parameters - args), and the way to compute the time functions
    (kwargs['ODE_model']). An internal computation is done to represent the parameters as a dictionary and feed the MSSM
    model, for example:
    param = {'C0': args[0], 'tao_c': args[1], 'alpha': args[2], 'C_0': args[3], 'V0': args[4], 'tao_v': args[5],
             'K_V': args[6], 'V_0': args[7], 'P0': args[8], 'k_NtV': args[9], 'k_Nt': args[10], 'tao_Nt': args[11],
             'Nt0': args[12], 'Nt_0': args[13] , 'k_EPSP': args[14], 'tao_EPSP': args[15],  'E_0': args[16]}
    :param time_vector: (numpy array [t, ]) time vector for simulation.
    :param args: (parameters) set of parameters for the MSSM.
    :param kwargs: (dictionary {'model': a,  'Input': b, 'params_name': c, 'mode': d, 'only spikes': e,
                                'block_ves_dep': f})
                   Extra parameters to specify:
                   a (MSSM_model) instance of MSSM_model class.
                   b (numpy array [t, ]) the input spike train.
                   c (list) the name of parameters. It must have the same size as *args, and it must  contain at least
                   one of the following names:
                     ['C0', 'tao_c', 'alpha', 'C_0', 'V0', 'tao_v', 'K_V', 'V_0', 'P0', 'k_NtV',
                     'k_Nt', 'tao_Nt', 'Nt0', 'Nt_0', 'k_EPSP', 'tao_EPSP', 'E_0']
                   d (String) the type of ODE solver supported by MSSM class. Up to now, there are two options:
                     'ODE': for Euler method.
                     'Analytical': for analytical solution of equations (no ODE solver).
                   e (boolean) if True, the output of the model is only the spike response of EPSP, otherwise the output
                     is the timeseries for EPSP.
                   f (boolean) if True, the model is not computed anymore if the weird behavior (due to the vesicle
                     depletion) is presented.
    :return: mssm.N (numpy array [t, ]) Output of neurotransmitters buffering.
    """
    # Local variables
    time_1 = m_time()
    params_name = kwargs['params_name']
    Input = kwargs['Input']
    output_model = None
    mssm = kwargs['model']
    param = {}
    return_only_spike_event = False
    output_spike_events = []
    L = len(time_vector)
    # print("utils, model_mssm, L ", L)
    # Creating param dictionary
    for i in range(len(args)):
        param[params_name[i]] = args[i]

    # Update parameters and initial conditions
    mssm.set_model_params(param)
    # mssm.set_initial_conditions()

    # If there is a neuron model
    model_neuron = None
    if "model_neuron" in kwargs:
        if kwargs['model_neuron'] is not None:
            model_neuron = kwargs['model_neuron']
            model_neuron.set_simulation_params()

    # Returning the entire time series or only the response after an input spike
    if "only spikes" in kwargs:
        # print("Utils.py, model_mssm(), only spikes is key in kwargs")
        return_only_spike_event = kwargs['only spikes']

    # Cutting the simulation of the MSSM if the condition of vesicle depletion (the weird behavior) happens
    block_sim_if_vesicle_depletion = False
    cond_cut_simulation = False
    if 'block_ves_dep' in kwargs:
        block_sim_if_vesicle_depletion = kwargs['block_ves_dep']

    time_init = m_time() - time_1

    # Evaluating MSSM model
    if kwargs['mode'] == "odeint":
        mssm.evaluate_odeint_time(time_vector, Input)
    else:
        ini_loop_time = m_time()
        for t in range(L):
            if kwargs['mode'] == "Analytical":
                mssm.evaluate_model_analytical(Input[t], t)
            elif kwargs['mode'] == "ODE":
                time_1 = m_time()
                mssm.evaluate_model_euler(Input[t], t)
                time_mssm = m_time() - time_1
            # elif kwargs['mode'] == "odeint":
            #     mssm.evaluate_odeint(t, [time_vector[t - 1], time_vector[t]], Input[t])
            else:
                assert False, "'ODE mode' must be either 'Analytical' or 'ODE'"

            # Considering the weird behavior of vesicles depletion to cut the simulation
            time_1 = m_time()
            if block_sim_if_vesicle_depletion:
                # If at least one synapse presents the vesicle depletion, then set flag to cut the simulation
                if len(mssm.ind_v_minor_to_0) > 0:
                    cond_cut_simulation = True

            # Evaluating neuron response
            if model_neuron is not None:
                model_neuron.update_state(mssm.EPSP[:, it], t)
                # Output based on the neuron model
                output_model = model_neuron.membrane_potential
            else:
                # Output based on the TM model
                output_model = mssm.EPSP
            time_extra = m_time() - time_1

            # Detecting spike events and storing model output
            time_1 = m_time()
            mssm.detect_spike_event(t, output_model)
            time_spike = m_time() - time_1
            # Cut simulation if the flag is True
            if cond_cut_simulation:
                break

            if t % 10000 == 0:
                # print("Value of t", t)
                # print_time(time_init, "time(init)")
                # print_time(time_mssm, "time(mssm)")
                # print_time(time_extra, "time(extra)")
                # print_time(time_spike, "time(spike event)")
                # print_time(m_time() - ini_loop_time, "Value of t " + str(t) + ". Loop")
                ini_loop_time = m_time()
        # Computing output spike event in the last ISI
        t = L
        spike_range = (mssm.time_spike_events[-1], t)
        mssm.compute_output_spike_event(spike_range, output_model)

    if return_only_spike_event:
        # return np.array(output_spike_events)
        return np.array(mssm.output_spike_events)
    if model_neuron is not None:
        return model_neuron.membrane_potential
    else:
        return mssm.EPSP


# **********************************************************************************************************************
# SPIKING NEURONS FUNCTIONS
def input_spike_train(sfreq, freq_signal, max_time, min_time=0.0):
    """
    spike train at a given frequency
    :param sfreq: sample frequency
    :param freq_signal: frequency of the output spike train
    :param max_time: Duration (in seconds) of the spike train
    :param min_time: time to start a spike train (in seconds)
    :return spike_train: Spike train
    """
    dt = 1 / sfreq  # size of steps from the sample frequency
    T = 1 / freq_signal  # Signal period
    step = int(T / dt)  # Where to generate an impulse
    L = int(max_time / dt)  # Number of samples in the desire time (max_time)
    spike_train = signal.unit_impulse(L, [i * step for i in range(int(np.ceil(L / step)))])
    if min_time > 0.0:
        spike_train[:int(min_time / dt) - 1] = 0.0
    return spike_train


# **********************************************************************************************************************
# ADDITIONAL FUNCTIONS
def m_time():
    return time.time()


def print_time(ts, msg):
    ms_res = ts * 1000
    min_res = ts / 60
    # print('Execution time:', ts, 'milliseconds')
    print(str(msg) + '. Execution time:', str(timedelta(seconds=ts)))


def loadObject(name, path='./'):
    """
    This function loads and returns an object, which should be located in path
    :param name: (String) name of the object-file
    :param path: (String)
    :return:
    res_object: (object)
    """
    pickleFile = open(path + name, 'rb')
    res_object = pickle.load(pickleFile)
    pickleFile.close()
    return res_object


def rmse(x, y):
    return np.square(np.subtract(x, y)).mean()


def su(params):
    return np.random.uniform(params[0], params[1])


def sn(params):
    return np.random.normal(params[0], params[1])


def plot_res_mssm(time_vector, input_signal, reference_signal, label):
    fig = plt.figure(figsize=(12, 2.8))
    plt.suptitle(label)
    ax = fig.add_subplot(121)
    ax.plot(time_vector, input_signal, c='gray')
    ax.set_xlabel("time(s)")
    ax.set_ylabel("Spikes")
    ax.set_title("Input Spike train")
    ax.grid()
    ax = fig.add_subplot(122)
    ax.plot(time_vector, reference_signal, c='gray')
    ax.set_xlabel("time(s)")
    ax.set_ylabel("A")
    ax.set_title("Postsynaptic response")
    ax.grid()
    fig.tight_layout(pad=0.5, w_pad=0.1, h_pad=1.0)


# **********************************************************************************************************************
# LOADING EXAMPLES OF FITTING
class Example_fitting:
    def __init__(self, model_str, path_signals=None, path_outputs=None):

        # Model
        self.model_str = model_str

        # Definition of time parameters
        self.r = None
        self.sfreq = None
        self.dt = None
        self.max_t = None
        self.end_t = None
        self.input_factor = None
        self.output_factor = None
        self.time_vector = None
        self.L = None

        # params dictionaries
        self.dict_params = None
        self.sim_params = None

        # parameters of paths
        self.path_signals = path_signals if path_signals is not None else path_signals_ext
        self.path_outputs = path_outputs if path_outputs is not None else path_outputs_ext

    def initial_params(self, ind, r_v=None, sfreq_v=None, max_t_v=None, end_t_v=None, input_factor_v=None,
                       output_factor_v=None, description=''):
        if r_v is None:
            r_v = r_ext
        if sfreq_v is None:
            sfreq_v = sfreq_ext
        if max_t_v is None:
            max_t_v = max_t_ext
        if end_t_v is None:
            end_t_v = end_t_ext
        if input_factor_v is None:
            input_factor_v = input_factor_ext
        if output_factor_v is None:
            output_factor_v = output_factor_ext

        self.r = r_v[ind]
        self.sfreq = sfreq_v[ind]
        self.dt = 1 / self.sfreq
        self.max_t = max_t_v[ind]
        self.end_t = end_t_v[ind]
        self.input_factor = input_factor_v[ind]
        self.output_factor = output_factor_v[ind]
        self.time_vector = np.arange(0, self.max_t, self.dt)
        self.L = self.time_vector.shape[0]

        # assigning dictionaries of parameters
        self.params_sim()
        self.params_dict(description=description)

    def params_dict(self, ext_par=None, description=''):
        dt = self.dt
        if self.model_str == 'MSSM':
            if ext_par is None:
                ext_pa = ext_par_mssm
            else:
                ext_pa = ext_par
            min_n = 2 * dt_f
            self.dict_params = {'model_str': self.model_str,
                                'params_name': params_name_mssm,
                                'bo': ((2 * dt, 0.0, 0.02, 2 * dt, 0.0, 1.0, 1e-1, dt, 1e-3, 2 * dt),
                                       (5.0, 10, 1e2, 1.0, 1.0, 1e2, max_kn, 5.0, 1e0, 10 * dt)),
                                'ini': [(9e-2, 3e-3), (5e-1, 3e-3), (5e0, 1e1), (5e-1, 3e-2), (1e-2, 1.0),
                                        (5e1, 1e1), (1e-1, 1e0), (min_n, 5), (5e-1, 3e-2), (2 * dt, 5 * dt)],
                                'type_sample_param': ['sn', 'sn', 'su', 'sn', 'su', 'sn', 'su', 'su', 'sn', 'su'],
                                'optimizer_mode': 'DE',
                                'ODE_mode': 'ODE',
                                'exp_ind_save': 0,
                                'only_spikes': False,
                                'path': self.path_outputs,
                                'description_file': description,
                                'output_factor': -self.output_factor,
                                'restriction_vesicle_depletion': False,  # True
                                }

        if self.model_str == 'TM':
            if ext_par is None:
                ext_pa = ext_par_tm
            else:
                ext_pa = ext_par
            self.dict_params = {'model_str': self.model_str,
                                'params_name': params_name_tm,
                                'bo': ((0.0, 2 * dt, 2 * dt, 0.0, 2 * dt),
                                       (1.0, 1.0, 1.0, 1e4, 10.0)),
                                'ini': [(0, 1), (2e-1, 3e-2), (2e-1, 3e-2), (0.0, 1e4), (2e-1, 3e-2)],
                                'type_sample_param': ['su', 'sn', 'sn', 'su', 'sn'],
                                'optimizer_mode': 'DE',
                                'ODE_mode': 'ODE',
                                'exp_ind_save': 0,
                                'only_spikes': False,
                                'path': self.path_outputs,
                                'description_file': description,
                                'output_factor': -self.output_factor,
                                'restriction_vesicle_depletion': False,
                                }

    def params_sim(self):
        self.sim_params = {'sfreq': self.sfreq, 'max_t': self.max_t, 'L': self.L, 'time_vector': self.time_vector}
