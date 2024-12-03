from synaptic_dynamic_models.SynDynModel import SynDynModel
import numpy as np


class TM_model(SynDynModel):
    def __init__(self, n_syn=1):
        super().__init__(n_syn)

        # model variables
        self.R = None
        self.U = None

        # Spike events
        self.R_spike_events = None
        self.U_spike_events = None
        self.R_steady_state = None
        self.U_steady_state = None

        # Output variables
        self.I_out = None

        # derivative variables
        self.d_R = None
        self.d_U = None

        # Params
        self.U0, self.tau_f, self.tau_d, self.Ase, self.tau_syn = None, None, None, None, None
        self.params = {'U0': 0.6, 'tau_f': 3e-3, 'tau_d': 800e-3, 'Ase': 250e-12, 'tau_syn': 20e-3}

        # Calling the main methods for setting the model
        self.set_model_params(self.params)
        self.set_simulation_params()

    def set_model_params(self, model_params):
        assert isinstance(model_params, dict), 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value
        self.U0, self.tau_f = self.params['U0'], self.params['tau_f']
        self.tau_d, self.Ase, self.tau_syn = self.params['tau_d'], self.params['Ase'], self.params['tau_syn']

    def set_initial_conditions(self, Input=None):
        # model variables
        self.R = np.zeros((self.n_syn, self.L))
        self.U = np.zeros((self.n_syn, self.L))

        # Output variables
        self.I_out = np.zeros((self.n_syn, self.L))

        # derivative variables
        self.d_R = np.zeros((self.n_syn, self.L))
        self.d_U = np.zeros((self.n_syn, self.L))

        # Variables for spike events and steady-state calculations
        self.R_spike_events = []
        self.U_spike_events = []
        self.output_spike_events = []
        self.time_spike_events = []
        self.R_steady_state = None
        self.U_steady_state = None
        self.output_steady_state = None
        self.efficacy = [0.0 for _ in range(self.n_syn)]
        self.efficacy_2 = [0.0 for _ in range(self.n_syn)]
        self.efficacy_3 = [0.0 for _ in range(self.n_syn)]
        self.t_steady_state = [0.0 for _ in range(self.n_syn)]
        self.t_max = [0.0 for _ in range(self.n_syn)]

        # Initial conditions
        self.R[:, 0] = 1.0
        aux_U0 = self.params['U0']
        if isinstance(aux_U0, np.ndarray):
            aux_U0 = self.params['U0'].reshape((-1, 1))
        self.U = np.multiply(np.ones((self.n_syn, self.L)), aux_U0)
        self.edge_detection = False

        if Input is None:
            self.Input = np.zeros((self.n_syn, self.L))
        else:
            assert isinstance(Input, np.ndarray), "'Input' must be a numpy array"
            assert len(spike_range) == 2, "'Input' must have 2-dimensions"
            self.Input = Input

    def evaluate_model_euler(self, I_it, it):
        self.Input[:, it] = I_it

        self.release_prob_u(I_it, it)
        self.resources_available_R(I_it, it)  # IMPORTANT TO UPDATE FIRST R(t)
        self.output(I_it, it)

    def get_output(self):
        return self.I_out

    def release_prob_u(self, I_it, it):
        dt = self.dt
        tau_f = self.tau_f
        U0 = self.U0
        # y0 = 0.0

        if it == 0:
            du = U0 * I_it  # -(dt / tau_f) * y0 + U0 * (1.0 - y0) * I_it
            self.U[:, it] = du
        else:
            du = -(dt / tau_f) * self.U[:, it - 1] + U0 * (1.0 - self.U[:, it - 1]) * I_it
            self.U[:, it] = self.U[:, it - 1] + du

    def resources_available_R(self, I_it, it):
        dt = self.dt
        tau_d = self.tau_d
        # x0 = 1.0

        if it == 0:
            dR = -self.U[:, 0] * I_it  # (dt / tau_d) * (1.0 - x0) - self.U[:, 0] * I_it
            self.R[:, 0] = self.R[:, 0] + dR
        else:
            dR = (dt / tau_d) * (1.0 - self.R[:, it - 1]) - self.U[:, it] * self.R[:, it - 1] * I_it
            self.R[:, it] = self.R[:, it - 1] + dR

    def output(self, I_it, it):
        dt = self.dt
        Ase = self.Ase
        tau_syn = self.tau_syn

        if it == 0:
            dI = -(dt / tau_syn) * self.I_out[:, it] + Ase * self.U[:, it] * I_it
            self.I_out[:, it] = self.I_out[:, it] + dI
        else:
            dI = -(dt / tau_syn) * self.I_out[:, it - 1] + Ase * self.R[:, it - 1] * self.U[:, it] * I_it
            self.I_out[:, it] = self.I_out[:, it - 1] + dI

    def append_spike_event(self, t, output):
        """
        Storing spike events for each state variable given a t-time
        Parameters
        ----------
        t
        output
        """
        self.R_spike_events.append(self.R[:, t])
        self.U_spike_events.append(self.U[:, t])
        # self.output_spike_events.append(self.I_out[:, t])
        self.time_spike_events.append(t)

        # Computing maximum EPSC response between the last and the current spike
        # If this is the first spike event
        if len(self.time_spike_events) > 1:
            spike_range = (self.time_spike_events[-2], self.time_spike_events[-1])
            self.compute_output_spike_event(spike_range, output)
