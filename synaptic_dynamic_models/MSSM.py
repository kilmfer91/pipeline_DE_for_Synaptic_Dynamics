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

        # Parameters
        self.C0, self.tau_c, self.alpha, self.V0, self.tau_v = None, None, None, None, None
        self.P0, self.k_NtV, self.k_Nt, self.tau_Nt, self.Nt0 = None, None, None, None, None
        self.k_EPSP, self.tau_EPSP, self.K_V, self.E_0 = None, None, None, None
        self.params = {'C0': 0.05, 'tau_c': 0.03, 'alpha': 1000, 'V0': 3.45, 'tau_v': 0.0084, 'P0': 0.5, 'k_NtV': 1,
                       'k_Nt': 1, 'tau_Nt': 0.008, 'Nt0': 0, 'k_EPSP': 10, 'tau_EPSP': 0.009, 'K_V': 1, 'E_0': 0}

        # Calling the main methods for setting the model
        self.set_model_params(self.params)
        self.set_simulation_params()
        self.set_initial_conditions()

    def set_model_params(self, model_params):
        """
        Set the paramateres for the MSSM
        :param model_params: (dict) parameters of MSSM. A dictionary with at least one of the following keys:
                             ['C0', 'tau_c', 'alpha', 'V0', 'tau_v', 'P0', 'k_NtV', 'k_Nt', 'tau_Nt', 'Nt0', 'k_EPSP',
                             'tau_EPSP', 'E_0']
        """
        assert isinstance(model_params, dict), 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value

        # Adjusting the initial P0
        self.params['C0'] = -np.log(1 - self.params['P0']) / self.params['V0']
        # Updating variables of the parameters
        self.C0, self.tau_c, self.alpha = self.params['C0'], self.params['tau_c'], self.params['alpha']
        self.V0, self.tau_v, self.P0 = self.params['V0'], self.params['tau_v'], self.params['P0']
        self.k_NtV, self.k_Nt, self.tau_Nt = self.params['k_NtV'], self.params['k_Nt'], self.params['tau_Nt']
        self.Nt0, self.k_EPSP, self.tau_EPSP = self.params['Nt0'], self.params['k_EPSP'], self.params['tau_EPSP']
        self.E_0 = self.params['E_0']
        # Adjusting parameters to make the system independent of the sampling frequency
        self.K_V = self.sim_params['sfreq']

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
        self.N[:, 0] = self.Nt0 / self.k_Nt
        self.P[:, 0] = self.P0
        self.V[:, 0] = self.V0
        self.C[:, 0] = self.C0
        self.EPSP[:, 0] = 0

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

    def evaluate_model_euler(self, I_t, t):
        """
        Compute the time functions of the MSSM by solving the ODE using the euler method
        :param I_t: (numpy array (n, t)) value of n-inputs at time t
        :param t: (int) time value
        """
        # Input
        self.Input[:, t] = I_t
        alpha = self.alpha / self.dt
        k_NtV = self.k_NtV / self.dt

        # model evaluation
        if t == 0:
            # Calcium buffering
            d_C_aux = (alpha * I_t) * self.dt
            self.C[:, t] = d_C_aux + self.C0
            # Probability of release
            self.P[:, 0] = I_t * self.P[:, 0]
            # Vesicle release
            dV_aux = (-self.K_V * (self.P0 * self.V0 * I_t)) * self.dt
            self.V[:, t] = dV_aux + self.V0
            # Neurotransmitter buffering
            dN_aux = k_NtV * -dV_aux * I_t * self.dt
            self.N[:, t] = dN_aux + self.Nt0
            # Excitatory Postsynaptic contribution
            d_EPSP_aux = self.k_EPSP * self.N[:, t] * self.dt / self.tau_EPSP
            self.EPSP[:, t] = d_EPSP_aux + self.E_0
        else:
            # Calcium buffering
            d_C_aux = (((self.C0 - self.C[:, t - 1]) / self.tau_c) + (alpha * I_t)) * self.dt
            self.C[:, t] = d_C_aux + self.C[:, t - 1]
            # Probability of release
            self.P[:, t] = I_t * (1 - np.exp(-self.C[:, t] * self.V[:, t - 1]))
            # self.P[:, t] = I_t * np.random.exponential(np.ones(self.n_syn))

            # Vesicle release
            factor_P_V = self.P[:, t] * self.V[:, t - 1] * I_t
            dV_aux = (((self.V0 - self.V[:, t - 1]) / self.tau_v) - self.K_V * factor_P_V) * self.dt
            self.V[:, t] = dV_aux + self.V[:, t - 1]
            # Neurotransmitter buffering
            dN_aux = (k_NtV * factor_P_V + ((self.Nt0 - self.k_Nt * self.N[:, t - 1]) / self.tau_Nt)) * self.dt
            self.N[:, t] = dN_aux + self.N[:, t - 1]
            # Excitatory Postsynaptic contribution
            d_EPSP_aux = (((self.E_0 - self.EPSP[:, t - 1]) + self.k_EPSP * self.N[:, t]) * (self.dt / self.tau_EPSP))
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
