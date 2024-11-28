from synaptic_dynamic_models.SynDynModel import SynDynModel
import numpy as np


class TM_model(SynDynModel):
    def __init__(self, n_syn=1):
        super().__init__(n_syn)

        # model variables
        self.x = None
        self.y = None
        self.z = None
        self.R = None
        self.U = None

        # Spike events
        self.x_spike_events = None
        self.y_spike_events = None
        self.z_spike_events = None
        self.R_spike_events = None
        self.U_spike_events = None
        self.x_steady_state = None
        self.y_steady_state = None
        self.z_steady_state = None
        self.R_steady_state = None
        self.U_steady_state = None

        # Output variables
        self.I_out = None
        self.I_out2 = None

        # derivative variables
        self.d_R = None
        self.d_U = None
        self.d_x = None
        self.d_y = None
        self.d_z = None

        self.params = {'U0': 0.6, 'x0': 1.0, 'y0': 0.0, 'z0': 1.0, 'tau_facil': None, 'tau_in': 3e-3, 'tau_rec': 800e-3,
                       'Ase': 250e-12, 'tau_syn': 20e-3}

        self.set_simulation_params()

    def set_model_params(self, model_params):
        assert type(model_params) != 'dict', 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value

    def set_initial_conditions(self, Input=None):
        # model variables
        self.x = np.zeros((self.n_syn, self.L))
        self.y = np.zeros((self.n_syn, self.L))
        self.z = np.zeros((self.n_syn, self.L))
        self.R = np.zeros((self.n_syn, self.L))
        self.U = np.zeros((self.n_syn, self.L))

        # Output variables
        self.I_out = np.zeros((self.n_syn, self.L))
        self.I_out2 = np.zeros((self.n_syn, self.L))

        # derivative variables
        self.d_R = np.zeros((self.n_syn, self.L))
        self.d_U = np.zeros((self.n_syn, self.L))
        self.d_x = np.zeros((self.n_syn, self.L))
        self.d_y = np.zeros((self.n_syn, self.L))
        self.d_z = np.zeros((self.n_syn, self.L))

        # Variables for spike events and steady-state calculations
        self.x_spike_events = []
        self.y_spike_events = []
        self.z_spike_events = []
        self.R_spike_events = []
        self.U_spike_events = []
        self.output_spike_events = []
        self.time_spike_events = []
        self.x_steady_state = None
        self.y_steady_state = None
        self.z_steady_state = None
        self.R_steady_state = None
        self.U_steady_state = None
        self.output_steady_state = None
        self.efficacy = [0.0 for _ in range(self.n_syn)]
        self.efficacy_2 = [0.0 for _ in range(self.n_syn)]
        self.efficacy_3 = [0.0 for _ in range(self.n_syn)]
        self.t_steady_state = [0.0 for _ in range(self.n_syn)]
        self.t_max = [0.0 for _ in range(self.n_syn)]

        # Initial conditions
        self.R[:, 0] = self.params['x0']
        self.x[:, 0] = self.params['x0']
        self.y[:, 0] = self.params['y0']
        self.z[:, 0] = self.params['z0']
        # If parameter tau_facil is None, then no facilitation dynamics (U constant along simulation as U_0)
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
        self.resources_available_R(I_it, it)  # IMPORTANT TO UPDATE FIRST Y (R)
        self.output2(I_it, it)

        # Detecting spike events
        # self.detect_spike_event(it)

    def get_output(self):
        return self.I_out2

    def release_prob_u(self, I_it, it):
        dt = self.dt
        tau_f = self.params['tau_in']
        U0 = self.params['U0']

        if it == 0:
            du = -(dt / tau_f) * self.params['y0'] + U0 * (1.0 - self.params['y0']) * I_it
            # print("t ", it, "du", du)
            self.U[:, it] = self.params['y0'] + du
        else:
            du = -(dt / tau_f) * self.U[:, it - 1] + U0 * (1.0 - self.U[:, it - 1]) * I_it
            self.U[:, it] = self.U[:, it - 1] + du

    def resources_available_R(self, I_it, it):
        dt = self.dt
        tau_d = self.params['tau_rec']

        if it == 0:
            dR = (dt / tau_d) * (1.0 - self.params['x0']) - self.U[:, 0] * self.params['x0'] * I_it
            self.R[:, 0] = self.R[:, 0] + dR
        else:
            dR = (dt / tau_d) * (1.0 - self.R[:, it - 1]) - self.U[:, it] * self.R[:, it - 1] * I_it
            self.R[:, it] = self.R[:, it - 1] + dR

    def output2(self, I_it, it):
        dt = self.dt
        Ase = self.params['Ase']
        tau_syn = self.params['tau_syn']

        if it == 0:
            dg = -(dt / tau_syn) * self.I_out2[:, it] + Ase * self.params['x0'] * self.U[:, it] * I_it
            # print("t ", it, "dg", dg)
            self.I_out2[:, it] = self.I_out2[:, it] + dg
            # self.I_out2[0] = self.params['Ase'] * self.U[0] * self.params['x0']
        else:
            dg = -(dt / tau_syn) * self.I_out2[:, it - 1] + Ase * self.R[:, it - 1] * self.U[:, it] * I_it
            self.I_out2[:, it] = self.I_out2[:, it - 1] + dg
            # self.I_out2[it] = self.params['Ase'] * self.U[it] * self.R[it - 1]

    def append_spike_event(self, t, output):
        """
        Storing spike events for each state variable given a t-time
        Parameters
        ----------
        t
        output
        """
        self.x_spike_events.append(self.x[:, t])
        self.y_spike_events.append(self.y[:, t])
        self.z_spike_events.append(self.z[:, t])
        self.R_spike_events.append(self.R[:, t])
        self.U_spike_events.append(self.U[:, t])
        # self.output_spike_events.append(self.I_out2[:, t])
        self.time_spike_events.append(t)

        # Computing maximum EPSP response between the last and the current spike
        # If this is the first spike event
        if len(self.time_spike_events) > 1:
            spike_range = (self.time_spike_events[-2], self.time_spike_events[-1])
            self.compute_output_spike_event(spike_range, output)