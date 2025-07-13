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
params_name_mssm = ['tau_c', 'alpha', 'V0', 'tau_v', 'P0', 'k_NtV', 'k_Nt', 'tau_Nt', 'k_EPSP', 'tau_EPSP']
ext_par_mssm = [3e-3, 905e-4, 3.45, 8.4e-3, 0.002, 1, 1, 13.5e-3, 10, 9e-3, 0.0, 0.0]  # From Karim C0 0.05
selected_params_mssm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

"""TM Model"""
# From the paper "Differential signaling via the same axon of neocortical pyramidal neurons" Markram, Wang, Tsodyks
params_name_tm = ['U0', 'tau_f', 'tau_d', 'Ase', 'tau_syn']
ext_par_tm = [0.03, 530e-3, 130e-3, 1540, 2.5e-3]
selected_params_tm = [0, 1, 2, 3, 4]

# labels for uploading examples of facilitation and depression
prefix_v = ['depression', 'facilitation']

# ******************************************************************************************************************
# TIME CONDITIONS
r_ext = [10, 30]  # Frequency of the spike train (Hz)
sfreq_ext = [2e3, 2e3]  # [2e3, 5e3]  # Sampling frequency (Hz)
ini_t_ext = [.1, .04]  # Initial time of simulation (s)
end_t_ext = [3.1, .55]  # Maximum time of spike train (s) (MUST BE LESS THAN max_t_v)
max_t_ext = [2.0, 1.2]  # [3.5, .8] # Maximum time of simulation (s)
# Input-Output factors
input_factor_ext = [1.0, 1.0]
output_factor_ext = [-1.0, -1.0]


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


def check_file(file):
    return os.path.isfile(file)


def rmse(x, y):
    if x.ndim != y.ndim:
        while x.ndim < y.ndim:
            x = np.expand_dims(x, axis=x.ndim)
        while y.ndim < x.ndim:
            y = np.expand_dims(y, axis=x.ndim)

    # aligning time dimension in last dimension
    if x.shape[-1] != y.shape[-1]:
        x = x.T

    # Computing RMSE in dim 1 (last dimension) for preserving the num of synapses
    return np.square(np.subtract(x, y)).mean(axis=1)


def su(params):
    return np.random.uniform(params[0], params[1])


def sn(params):
    return np.random.normal(params[0], params[1])


def plot_temp_mssm(model_stp, fig, titleSize=12, labelSize=9, ind_interest=0):
    time_vector = model_stp.time_vector
    output = model_stp.get_output()

    ax = fig.add_subplot(231)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.C, axis=0), np.max(model_stp.C, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.C, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.C[ind_interest, :])
    ax.set_title(r'C(t) [$\mu$M]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', c='gray', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(232)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.V, axis=0), np.max(model_stp.V, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.V, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.C[ind_interest, :])
    ax.set_title(r'V(t) [$\mu$M]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel(r'', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(233)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.P, axis=0), np.max(model_stp.P, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.P, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.P[ind_interest, :])
    ax.set_title('$P(t)$', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(234)
    if ind_interest is None:
        ax.fill_between(time_vector, np.min(model_stp.N, axis=0), np.max(model_stp.N, axis=0), color="darkgrey",
                        alpha=0.3)
        ax.plot(time_vector, np.mean(model_stp.N, axis=0), c='black')
    else:
        ax.plot(time_vector, model_stp.N[ind_interest, :])
    ax.set_title(r'$N(t)$ [$\mu$M]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel('$\mu$M')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    ax = fig.add_subplot(235)
    if ind_interest is None:
        ax.plot(time_vector, np.mean(output, axis=0), c='black')
        ax.fill_between(time_vector, np.min(output, axis=0), np.max(output, axis=0), color="darkgrey",
                        alpha=0.3)
    else:
        ax.plot(time_vector, output[ind_interest, :])
    ax.set_title('$EPSP(t)$ [mV]', fontsize=titleSize)
    ax.set_xlabel('time (s)', c='gray', fontsize=labelSize)
    # ax.set_ylabel('$\mu$M')
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(8)
    ax.grid()

    fig.subplots_adjust(top=0.92)
    # Final adjustments
    fig.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


# **********************************************************************************************************************
# LOADING EXAMPLES OF FITTING
class Example_fitting:
    def __init__(self, model_str, path_signals=None, path_outputs=None):

        # Model
        self.model_str = model_str

        # Definition of time parameters
        self.sfreq = None
        self.dt = None
        self.max_t = None
        self.input_factor = None
        self.output_factor = None
        self.time_vector = None
        self.L = None
        self.r = None

        # params dictionaries
        self.dict_params = None
        self.sim_params = None
        self.DE_params = []

        # parameters of paths
        self.path_signals = path_signals if path_signals is not None else path_signals_ext
        self.path_outputs = path_outputs if path_outputs is not None else path_outputs_ext

    def initial_params(self, ind, *args_):
        """
        args =
        :param ind:
        :param args_: tuple() [sfreq_v, max_t_v, input_factor_v, output_factor_v, description, r_v]
        :return:
        """
        args = args_[0]

        if args[0] is None:
            sfreq_v = sfreq_ext
        if args[1] is None:
            max_t_v = max_t_ext
        if args[2] is None:
            input_factor_v = input_factor_ext
        if args[3] is None:
            output_factor_v = output_factor_ext
        if args[5] is None:
            r_v = r_ext

        self.sfreq = args[0]
        self.dt = 1 / self.sfreq
        self.max_t = args[1]
        self.input_factor = args[2]
        self.output_factor = args[3]
        self.time_vector = np.arange(0, self.max_t, self.dt)
        self.L = self.time_vector.shape[0]
        self.r = args[5]

        # assigning dictionaries of parameters
        self.params_sim()
        self.params_dict(description=args[4])

    def params_DE(self, strategy='best1bin', generations=1000, popsize=15, tol=0.01, mutation=(0.5, 1),
                  recombination=0.7, seed=None, callback=None, disp=True, polish=True, init='latinhypercube', atol=0,
                  updating='immediate', workers=1, constraints=(), x0=None, integrality=None, vectorized=False):
        self.DE_params.append(strategy)
        self.DE_params.append(generations)
        self.DE_params.append(popsize)
        self.DE_params.append(tol)
        self.DE_params.append(mutation)
        self.DE_params.append(recombination)
        self.DE_params.append(seed)
        self.DE_params.append(callback)
        self.DE_params.append(disp)
        self.DE_params.append(polish)
        self.DE_params.append(init)
        self.DE_params.append(atol)
        self.DE_params.append(updating)
        self.DE_params.append(workers)
        self.DE_params.append(constraints)
        self.DE_params.append(x0)
        self.DE_params.append(integrality)
        self.DE_params.append(vectorized)

    def params_dict(self, ext_par=None, description=''):
        dt = self.dt
        if self.model_str == 'MSSM':
            if ext_par is None:
                ext_pa = ext_par_mssm
            else:
                ext_pa = ext_par
            min_n = 2 * dt
            self.dict_params = {'model_str': self.model_str,
                                'params_name': params_name_mssm,
                                'bo': ((2 * dt, 0.0, 0.02, 2 * dt, 0.0, 1.0, 1e-1, 2 * dt,  1e-3, 2 * dt),
                                       (0.1,    10,  1e2,  1.0,    1.0, 1e2, 1.0,  10 * dt, 1e0,  10 * dt)),
                                'ODE_mode': 'ODE',
                                'ind_experiment': 0,
                                'only_spikes': False,
                                'path': self.path_outputs,
                                'description_file': description,
                                'output_factor': self.output_factor,
                                'frequency_reference': False,
                                'ref_freq_vector': None,
                                }

        if self.model_str == 'TM':
            if ext_par is None:
                ext_pa = ext_par_tm
            else:
                ext_pa = ext_par
            self.dict_params = {'model_str': self.model_str,
                                'params_name': params_name_tm,
                                'bo': ((0.0, 2 * dt, 2 * dt, 0.0, 2 * dt),
                                       (1.0, 1.0,    1.0,    1e4, 10.0)),
                                'ODE_mode': 'ODE',
                                'ind_experiment': 0,
                                'only_spikes': False,
                                'path': self.path_outputs,
                                'description_file': description,
                                'output_factor': self.output_factor,
                                'frequency_reference': False,
                                }

    def params_sim(self):
        self.sim_params = {'sfreq': self.sfreq, 'max_t': self.max_t, 'L': self.L, 'time_vector': self.time_vector}
