from synaptic_dynamic_models.SynDynModel import SynDynModel
import numpy as np
from scipy.integrate import odeint


class MSSM_model(SynDynModel):

    def __init__(self, n_syn=1):
        super().__init__(n_syn)

        # model variables
        self.C = None
        self.V = None
        self.N = None
        self.P = None
        self.EPSP = None

        # Spike events
        self.C_spike_events = None
        self.V_spike_events = None
        self.N_spike_events = None
        self.P_spike_events = None
        self.C_steady_state = None
        self.V_steady_state = None
        self.N_steady_state = None
        self.P_steady_state = None

        # derivative variables
        self.d_N = None
        self.d_C = None
        self.d_V = None
        self.d_EPSP = None

        # variables for analytical funtions
        self.impulse_response_c = None
        self.impulse_response_v = None
        self.impulse_response_n = None
        self.impulse_response_e = None

        # Variable for detecting weird behavior of MSSM with vesicles
        self.ind_v_minor_to_0 = None

        self.params = {'C0': 0.05, 'tao_c': 0.03, 'alpha': 1000, 'V0': 3.45, 'tao_v': 0.0084, 'P0': 0.5, 'k_NtV': 1,
                       'k_Nt': 1, 'tao_Nt': 0.008, 'Nt0': 0, 'k_EPSP': 10, 'tao_EPSP': 0.009, 'C_0': 0.015, 'V_0': 1,
                       'Nt_0': 0, 'K_V': 1, 'E_0': 0}

        self.set_simulation_params()
        self.set_initial_conditions()

    def set_initial_conditions(self, Input=None):
        """
        Setting initial values for all variables
        """
        # state variables
        self.C = np.zeros((self.n_syn, self.L))
        self.d_C = np.zeros((self.n_syn, self.L))
        self.N = np.zeros((self.n_syn, self.L))
        self.d_N = np.zeros((self.n_syn, self.L))
        self.V = np.zeros((self.n_syn, self.L))
        self.d_V = np.zeros((self.n_syn, self.L))
        self.P = np.zeros((self.n_syn, self.L))
        self.EPSP = np.zeros((self.n_syn, self.L))
        self.d_EPSP = np.zeros((self.n_syn, self.L))

        # Initial conditions
        self.params['C_0'] = self.params['C0']
        self.params['Nt_0'] = self.params['Nt0'] / self.params['k_Nt']
        self.params['V_0'] = self.params['V0']
        self.N[:, 0] = self.params['Nt_0']
        self.P[:, 0] = self.params['P0']
        self.V[:, 0] = self.params['V_0']
        self.C[:, 0] = self.params['C_0']
        self.EPSP[:, 0] = 0

        # Adjusting parameters to make the system independent of the sampling frequency
        # self.params['alpha'] = self.params['alpha'] / self.dt
        # self.params['k_NtV'] = self.params['k_NtV'] / self.dt
        self.params['K_V'] = self.sim_params['sfreq']  # = 1 / dt

        # variables for analytical solutions
        self.impulse_response_c = [[] for _ in range(self.n_syn)]
        self.impulse_response_v = [[] for _ in range(self.n_syn)]
        self.impulse_response_n = [[] for _ in range(self.n_syn)]
        self.impulse_response_e = [[] for _ in range(self.n_syn)]

        # Variables for spike events and steady-state calculations
        self.C_spike_events = []
        self.V_spike_events = []
        self.N_spike_events = []
        self.P_spike_events = []
        self.output_spike_events = []
        self.time_spike_events = []
        self.C_steady_state = None
        self.V_steady_state = None
        self.N_steady_state = None
        self.P_steady_state = None
        self.output_steady_state = None
        self.efficacy = [0.0 for _ in range(self.n_syn)]
        self.efficacy_2 = [0.0 for _ in range(self.n_syn)]
        self.efficacy_3 = [0.0 for _ in range(self.n_syn)]
        self.t_steady_state = [0.0 for _ in range(self.n_syn)]
        self.t_max = [0.0 for _ in range(self.n_syn)]
        self.edge_detection = False
        if Input is None:
            self.Input = np.zeros((self.n_syn, self.L))
        else:
            assert isinstance(Input, np.ndarray), "'Input' must be a numpy array"
            assert len(spike_range) == 2, "'Input' must have 2-dimensions"
            self.Input = Input

        # Variable for detecting weird behavior of MSSM with vesicles
        self.ind_v_minor_to_0 = []

    def set_model_params(self, model_params):
        """
        Set the paramateres for the MSSM
        :param model_params: (dict) parameters of MSSM. A dictionary with at least one of the following keys:
                             ['C0', 'tao_c', 'alpha', 'V0', 'tao_v', 'P0', 'k_NtV', 'k_Nt', 'tao_Nt', 'Nt0', 'k_EPSP',
                             'tao_EPSP', 'C_0', 'V_0', 'Nt_0', 'K_V', 'E_0']
        """
        assert type(model_params) != 'dict', 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value

        # Adjusting the initial P0
        self.params['C0'] = -np.log(1 - self.params['P0']) / self.params['V0']
        self.params['C_0'] = -np.log(1 - self.params['P0']) / self.params['V_0']

        self.set_initial_conditions()

    def evaluate_model_euler(self, I_t, t):
        """
        Compute the time functions of the MSSM by solving the ODE using the euler method
        :param I_t: (numpy array (n, t)) value of n-inputs at time t
        :param t: (int) time value
        """
        # print("Evaluating euler method for class MSSM")
        # Input
        self.Input[:, t] = I_t
        alpha = self.params['alpha'] / self.dt
        k_NtV = self.params['k_NtV'] / self.dt

        # model evaluation
        if t == 0:
            # Calcium buffering
            d_C_aux = (alpha * I_t) * self.dt
            self.C[:, t] = d_C_aux + self.params['C_0']
            # Probability of release
            self.P[:, 0] = I_t * self.P[:, 0]
            # Vesicle release
            dV_aux = (-self.params['K_V'] * (self.params['P0'] * I_t)) * self.dt
            self.V[:, t] = dV_aux + self.params['V_0']
            # Neurotransmitter buffering
            # dV = self.V[:, t] - self.params['V_0']
            dN_aux = k_NtV * -dV_aux * I_t * self.dt
            self.N[:, t] = dN_aux + self.params['Nt_0']
            # Excitatory Postsynaptic contribution
            d_EPSP_aux = self.params['k_EPSP'] * self.N[:, t] * self.dt / self.params['tao_EPSP']
            self.EPSP[:, t] = d_EPSP_aux + self.params['E_0']
        else:
            # Calcium buffering
            d_C_aux = (((self.params['C0'] - self.C[:, t - 1]) / self.params['tao_c']) + (alpha * I_t)) * self.dt
            self.C[:, t] = d_C_aux + self.C[:, t - 1]
            # Probability of release
            self.P[:, t] = I_t * (1 - np.exp(-self.C[:, t - 1] * self.V[:, t - 1])) # 1 - np.exp(-self.C[:, t - 1] * self.V[:, t - 1])  #
            # Vesicle release
            dV_aux = (((self.params['V0'] - self.V[:, t - 1]) / self.params['tao_v']) - self.params['K_V'] * (self.P[:, t] * I_t)) * self.dt
            self.V[:, t] = np.clip(dV_aux + self.V[:, t - 1], 0, None)
            # self.V[:, t] = dV_aux + self.V[:, t - 1]
            if I_t > 0 and np.any(self.V[:, t] <= 0):
                self.ind_v_minor_to_0 = np.where(self.V[:, t] <= 0)[0]
            # Neurotransmitter buffering
            # dV = self.V[:, t] - self.V[:, t - 1]  # self.d_V[:, t]
            dN_aux = (k_NtV * -dV_aux * I_t + ((self.params['Nt0'] - self.params['k_Nt'] * self.N[:, t - 1]) / self.params['tao_Nt'])) * self.dt
            self.N[:, t] = dN_aux + self.N[:, t - 1]
            # Excitatory Postsynaptic contribution
            d_EPSP_aux = ((self.params['E_0'] - self.EPSP[:, t - 1]) + self.params['k_EPSP'] * self.N[:, t]) * (self.dt / self.params['tao_EPSP'])
            self.EPSP[:, t] = d_EPSP_aux + self.EPSP[:, t - 1]

    def get_output(self):
        return self.EPSP

    def append_spike_event(self, t, output):
        """
        Storing spike events for each state variable given a t-time
        Parameters
        ----------
        t
        output
        """
        self.C_spike_events.append(self.C[:, t])
        self.V_spike_events.append(self.V[:, t])
        self.N_spike_events.append(self.N[:, t])
        self.P_spike_events.append(self.P[:, t])
        # self.output_spike_events.append(self.EPSP[:, t])
        self.time_spike_events.append(t)

        # Computing maximum EPSP response between the last and the current spike
        # If this is the first spike event
        if len(self.time_spike_events) > 1:
            spike_range = (self.time_spike_events[-2], self.time_spike_events[-1])
            self.compute_output_spike_event(spike_range, output)