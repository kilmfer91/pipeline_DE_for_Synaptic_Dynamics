from synaptic_dynamic_models.SynDynModel import SynDynModel
import numpy as np
from scipy.integrate import odeint


class LAP_model(SynDynModel):

    def __init__(self, n_syn=1):
        super().__init__(n_syn)

        # Model variables
        self.Cai = None
        self.Rrel = None
        self.FluxGlu = None
        self.Prel = None
        self.Krecov = None
        self.EPSC = None

        # Spike events
        self.Cai_spike_events = None
        self.Rrel_spike_events = None
        self.FluxGlu_spike_events = None
        self.Prel_spike_events = None
        self.Krecov_spike_events = None
        self.Cai_steady_state = None
        self.Rrel_steady_state = None
        self.FluxGlu_steady_state = None
        self.Prel_steady_state = None
        self.Krecov_steady_state = None

        # derivative variables
        self.d_Glu = None
        self.d_Cai = None
        self.d_Rrel = None
        self.d_EPSC = None
        self.d_EPSC_aux = None

        # Steady-state variables
        self.Cai_ss = None
        self.Krecov_ss = None
        self.Prel_ss = None
        self.Rrel_ss = None
        self.Fres = None
        self.Fres2 = None

        # variables for analytical funtions
        self.impulse_response_c = None
        self.impulse_response_r = None
        self.impulse_response_n = None
        self.impulse_response_e = None

        # Parameters
        self.params = {'Cai_0': 7.5, 'KCa': 515, 'Krel_half': 20, 'Krecov_0': 7.5e-3, 'Krecov_max': 7.5e-3,
                       'Prel0': 0.02, 'Prel_max': 1.0, 'tao_Cai': 0.450, 'n': 1, 'Ntotal': 9.5e6, 'nHill': 4,
                       'Krecov_half': 20, 'KGlu': 8.0e-3, 'tao_EPSC': 5.0e-3, 'Krel': 1, 'Krel2': 1,
                       'Cai0': 1.0, 'Rrel0': 1.0}

        self.set_simulation_params()
        self.set_initial_conditions()

    def set_initial_conditions(self, Input=None):
        # state variables
        self.Cai = np.zeros((self.n_syn, self.L))
        self.d_Cai = np.zeros((self.n_syn, self.L))
        self.FluxGlu = np.zeros((self.n_syn, self.L))
        self.d_Glu = np.zeros((self.n_syn, self.L))
        self.Rrel = np.zeros((self.n_syn, self.L))
        self.d_Rrel = np.zeros((self.n_syn, self.L))
        self.Prel = np.zeros((self.n_syn, self.L))
        self.Krecov = np.zeros((self.n_syn, self.L))
        self.EPSC = np.zeros((self.n_syn, self.L))
        self.d_EPSC = np.zeros((self.n_syn, self.L))

        # variables for analytical solutions
        self.impulse_response_c = [[] for _ in range(self.n_syn)]
        self.impulse_response_r = [[] for _ in range(self.n_syn)]
        self.impulse_response_n = [[] for _ in range(self.n_syn)]
        self.impulse_response_e = [[] for _ in range(self.n_syn)]

        # Variables for spike events and steady-state calculations
        self.Cai_spike_events = []
        self.Rrel_spike_events = []
        self.FluxGlu_spike_events = []
        self.Prel_spike_events = []
        self.Krecov_spike_events = []
        self.output_spike_events = []  # [[] for _ in range(self.n_syn)][[] for _ in range(self.n_syn)]
        self.time_spike_events = []  # [[] for _ in range(self.n_syn)]
        self.Cai_steady_state = None
        self.Rrel_steady_state = None
        self.FluxGlu_steady_state = None
        self.Prel_steady_state = None
        self.Krecov_steady_state = None
        self.output_steady_state = None
        self.efficacy = [0.0 for _ in range(self.n_syn)]
        self.efficacy_2 = [0.0 for _ in range(self.n_syn)]
        self.efficacy_3 = [0.0 for _ in range(self.n_syn)]
        self.t_steady_state = [0.0 for _ in range(self.n_syn)]
        self.t_max = [0.0 for _ in range(self.n_syn)]

        self.Cai[:, 0] = self.params['Cai_0']
        self.FluxGlu[:, 0] = [0.0 for _ in range(self.n_syn)]
        self.Rrel[:, 0] = self.params['Rrel0']
        self.Prel[:, 0] = self.params['Prel0']
        self.Krecov[:, 0] = self.params['Krecov_0']
        self.EPSC[:, 0] = [0.0 for _ in range(self.n_syn)]
        self.edge_detection = False

        if Input is None:
            self.Input = np.zeros((self.n_syn, self.L))
        else:
            assert isinstance(Input, np.ndarray), "'Input' must be a numpy array"
            assert len(spike_range) == 2, "'Input' must have 2-dimensions"
            self.Input = Input

        # r_resonance = r_resonance_v[ind]
        # r = r_v[ind]
        Glu = 0
        Krecov_ss = 0
        Np = 0
        Nrel = 0
        Nempty = 0

    def set_model_params(self, model_params):
        assert type(model_params) != 'dict', 'params should be a dict'
        for key, value in model_params.items():
            if key in self.params.keys():
                self.params[key] = value

        # Adjusting the initial P0
        self.params['Cai_0'] = np.power((np.power(self.params['Krel_half'], self.params['nHill']) *
                                         self.params['Prel0']) / (
                                                    self.params['Prel_max'] - self.params['Prel0']),
                                        1 / self.params['nHill'])
        self.params['Krel'] = self.sim_params['sfreq']

        self.set_initial_conditions()

    def evaluate_model_euler(self, ICa_it, it):
        self.Input[:, it] = ICa_it
        self.calcium_concentration(ICa_it, it)
        self.release_probability(it)
        self.vesicle_release(ICa_it, it)
        self.glutamate_flow(ICa_it, it)
        self.compute_EPSC(it)

        # Detecting spike events
        # self.detect_spike_event(it)

    def model_lap_odeint(self, ini_cond, t, u, ti):
        # Initial conditions
        Cai = ini_cond[0]
        Krecov = ini_cond[1]
        Prel = ini_cond[2]
        Rrel = ini_cond[3]
        FluxGlu = ini_cond[4]
        EPSC = ini_cond[5]

        # Calcium buffering
        Cai_0 = self.params['Cai_0']
        KCa = self.params['KCa'] / (1000 * self.dt)
        tao_Cai = self.params['tao_Cai']  # * self.dt
        # K recovery
        Krecov_0 = self.params['Krecov_0'] * 1000  # / self.dt
        Krecov_max = self.params['Krecov_max'] * 1000  # / self.dt
        Krecov_half = self.params['Krecov_half']
        # Vesicle release
        # Krel2 = self.params['Krel2']
        Krel = self.params['Krel']
        # Release probability
        Prel_max = self.params['Prel_max']
        nHill = self.params['nHill']
        Krel_half = self.params['Krel_half']
        # Glutamate flux
        n = self.params['n']
        Ntotal = self.params['Ntotal']
        # EPSC
        tao_EPSC = self.params['tao_EPSC']  # * self.dt
        KGlu = self.params['KGlu'] / self.dt

        # Differential equations
        dCdt = (Cai_0 - Cai + KCa * u) / tao_Cai
        Krecov = Krecov_0 + ((Krecov_max - Krecov_0) * (Cai / (Cai + Krecov_half)))  # (Cai + Krecov)))
        Prel = Prel_max * (np.power(Cai, nHill) / (np.power(Cai, nHill) + np.power(Krel_half, nHill)))
        # dRreldt = Krel2 * Krecov * (1 - Rrel) - Krel * Prel * Rrel * u
        # dRreldt = Krel * (Krecov * (1 - Rrel) - Prel * Rrel * u)
        dRreldt = Krecov * (1 - Rrel) - Krel * Prel * Rrel * u
        FluxGlu = (n * Ntotal * Rrel * Prel * u)
        dEPSCdt = ((-EPSC - KGlu * FluxGlu) / tao_EPSC)
        return [dCdt, Krecov, Prel, dRreldt, FluxGlu, dEPSCdt]

    def evaluate_odeint(self, t, tspan, Input):

        dt = self.dt
        Krel = self.params['Krel']
        Krel2 = self.params['Krel2']
        # K recovery
        Krecov_0 = self.params['Krecov_0'] * 1000  # / self.dt
        Krecov_max = self.params['Krecov_max'] * 1000  # / self.dt
        Krecov_half = self.params['Krecov_half']
        # Release probability
        Prel_max = self.params['Prel_max']
        nHill = self.params['nHill']
        Krel_half = self.params['Krel_half']
        # Glutamate flux
        n = self.params['n']
        Ntotal = self.params['Ntotal']
        # Calcium concentration
        Cai_0 = self.params['Cai_0']
        KCa = self.params['KCa'] / (1000 * self.dt)
        tao_Cai = self.params['tao_Cai']
        # EPSC
        KGlu = self.params['KGlu'] / self.dt
        tao_EPSC = self.params['tao_EPSC']

        # Response of the system for a spike in t=0
        if Input == 1 and t == 0:
            d_C_aux = (KCa * Input / tao_Cai) * self.dt
            self.Cai[0] = d_C_aux + Cai_0
            # self.Cai[0] += self.impulse_response(t, KCa * dt, 1 / tao_Cai, dt / tao_Cai, t)
            self.Krecov[0] = Krecov_0 + ((Krecov_max - Krecov_0) * (self.Cai[0] / (self.Cai[0] + Krecov_half)))
            self.Prel[0] = Prel_max * (np.power(self.Cai[0], nHill) / (np.power(self.Cai[0], nHill) +
                                                                        np.power(Krel_half, nHill)))
            d_Rrel_aux = (-Krel * self.Prel[0] * self.Rrel[0] * Input) * self.dt
            self.Rrel[0] = d_Rrel_aux + self.Rrel[0]
            # self.Rrel[0] = 1 - self.impulse_response(t, self.Rrel[0], self.Prel[0] * Krel * dt, self.Krecov[0], t)
            self.FluxGlu[0] = (n * Ntotal * self.Rrel[0] * self.Prel[0] * Input)
            d_EPSC_aux = ((-KGlu * self.FluxGlu[0]) / tao_EPSC) * self.dt
            self.EPSC[0] = d_EPSC_aux + 0
            # self.EPSC[0] -= self.impulse_response(t, KGlu * self.FluxGlu[0] * dt, 1 / tao_EPSC, dt / tao_EPSC, t)

        if t > 0:
            # loop through the synapses IMPROVE THIS TO VECTORIZE THE COMPUTATION OF ODEINT
            for s in range(self.n_syn):
                #  initial conditions
                ini_cond = [self.Cai[s, t - 1], self.Krecov[s, t - 1], self.Prel[s, t - 1],
                            self.Rrel[s, t - 1], self.FluxGlu[s, t - 1], self.EPSC[s, t - 1]]
                #  Compute ODE solver
                z = odeint(self.model_lap_odeint, ini_cond, tspan, args=(Input, t))
                # store solutions
                self.Cai[s, t] = z[1][0]
                self.Rrel[s, t] = z[1][3]
                self.EPSC[s, t] = z[1][5]
                self.Krecov[s, t] = Krecov_0 + ((Krecov_max - Krecov_0) * (self.Cai[s, t] / (self.Cai[s, t] + Krecov_half)))
                self.Prel[s, t] = Prel_max * (np.power(self.Cai[s, t], nHill) /
                                           (np.power(self.Cai[s, t], nHill) + np.power(Krel_half, nHill)))
                self.FluxGlu[s, t] = (n * Ntotal * self.Rrel[s, t] * self.Prel[s, t] * Input)

        # Detecting spike events
        # self.detect_spike_event(t)

    def get_output(self):
        return self.EPSC

    def calcium_concentration(self, ICa_it, it):
        """
        Calcium contrentration
        :param ICa_it:
        :param it:
        """
        Cai_0 = self.params['Cai_0']
        Cai = self.Cai[:, it]
        tao_Cai = self.params['tao_Cai']
        # KCa = self.params['KCa']
        KCa = self.params['KCa'] / (1000 * self.dt)

        # New initial condition whether there is Input in t=0 or not
        if it == 0:
            Cai = Cai_0
            d_C_aux = (KCa * ICa_it / tao_Cai) * self.dt
            self.Cai[:, it] = d_C_aux + Cai
            self.d_Cai[:, it] = 0

        # Dynamic of Calcium buffering
        if it > 0:
            Cai = self.Cai[:, it - 1]
            d_Cai_aux = ((Cai_0 - Cai + KCa * ICa_it) / tao_Cai) * self.dt
            self.Cai[:, it] = np.clip(d_Cai_aux + Cai, 0, None)
            self.d_Cai[:, it] = d_Cai_aux

        # if self.Cai[it] < 0:
        #     self.Cai[it] = 0

    def k_recovery(self, I_t, t):
        """
        Computing k-recovery
        :param I_t:
        :param t:
        """
        # Local variables
        Krecov_0 = self.params['Krecov_0'] / self.dt
        Krecov_max = self.params['Krecov_max'] / self.dt
        Krecov_half = self.params['Krecov_half']
        Cai = self.Cai[t]
        # Cai = self.Cai[t - 1]

        self.Krecov[t] = Krecov_0 + ((Krecov_max - Krecov_0) * (Cai / (Cai + Krecov_half)))  # (Cai + Krecov)))

    def vesicle_release(self, ICa_it, it):
        """
        # Remaining ratio of Vesicle release. ADDING Krel and Krel2 FROM MY SIDE
        :param ICa_it:
        :param it:
        """

        # """
        # Local variables
        Krel = self.params['Krel']
        Krecov_0 = self.params['Krecov_0'] * 1000  # / self.dt
        Krecov_max = self.params['Krecov_max'] * 1000  # / self.dt
        Krecov_half = self.params['Krecov_half']
        Cai = self.Cai[:, it]

        # New initial condition whether there is Input in t=0 or not
        if it == 0:
            Prel = self.params['Prel0']
            Rrel = self.params['Rrel0']
            d_Rrel_aux = (-Krel * Prel * Rrel * ICa_it) * self.dt
            self.Rrel[:, it] = d_Rrel_aux + Rrel
            self.d_Rrel[:, it] = 0

        # Dynamic of Vesicle release
        if it > 0:
            Prel = self.Prel[:, it - 1]
            Rrel = self.Rrel[:, it - 1]
            # Krecov = self.Krecov[it - 1]
            Krecov = Krecov_0 + ((Krecov_max - Krecov_0) * (Cai / (Cai + Krecov_half)))  # (Cai + Krecov)))
            self.Krecov[:, it] = Krecov
            # self.Krecov[it] = Krecov_0 + ((Krecov_max - Krecov_0) * (Cai / (Cai + Krecov_half)))  # (Cai + Krecov)))
            # d_Rrel_aux = (Krel2 * Krecov * (1 - Rrel) - Krel * Prel * Rrel * ICa_it) * self.dt
            d_Rrel_aux = (Krecov * (1 - Rrel) - Krel * Prel * Rrel * ICa_it) * self.dt
            self.Rrel[:, it] = np.clip(d_Rrel_aux + Rrel, 0, None)
            self.d_Rrel[:, it] = d_Rrel_aux

        # if self.Rrel[it] < 0:
        #     self.Rrel[it] = 0

    def release_probability(self, it):
        """
        Probability of neurotransmitters release
        """
        Prel_max = self.params['Prel_max']
        nHill = self.params['nHill']
        Krel_half = self.params['Krel_half']
        # Cai = self.Cai[it]
        # Probability of release
        # self.Prel[it + 1] = Prel_max * (np.power(Cai, nHill) / (np.power(Cai, nHill) + np.power(Krel_half, nHill)))

        # if it == 0:
        #     self.Prel[it] = self.params['Prel0']
        # if it > 0:
            # Cai = self.Cai[it - 1]
        Cai = self.Cai[:, it]

        # Probability of release
        self.Prel[:, it] = Prel_max * (np.power(Cai, nHill) / (np.power(Cai, nHill) + np.power(Krel_half, nHill)))

    def glutamate_flow(self, ICa_it, it):
        """
        Flow of Glutamate
        """
        n = self.params['n']
        Ntotal = self.params['Ntotal']
        if it == 0:
            Prel = self.params['Prel0']
            Rrel = self.Rrel[:, it]
        else:
            Prel = self.Prel[:, it - 1]
            Rrel = self.Rrel[:, it]

        self.FluxGlu[:, it] = (n * Ntotal * Rrel * Prel * ICa_it)

    def compute_EPSC(self, it):
        """
        Excitatory Postsynaptic current
        :param it:
        """
        tao_EPSC = self.params['tao_EPSC']  # * self.dt
        KGlu = self.params['KGlu'] / self.dt

        # New initial condition whether there is Input in t=0 or not
        if it == 0:
            EPSC = 0
            FluxGlu = self.FluxGlu[:, it]
            d_EPSC_aux = ((-EPSC - KGlu * FluxGlu) / tao_EPSC) * self.dt
            self.EPSC[:, it] = d_EPSC_aux + EPSC
            self.d_EPSC[:, it] = d_EPSC_aux

        if it > 0:
            # FluxGlu = self.FluxGlu[:, it - 1]
            FluxGlu = self.FluxGlu[:, it]
            EPSC = self.EPSC[:, it - 1]
            d_EPSC_aux = ((-EPSC - KGlu * FluxGlu) / tao_EPSC) * self.dt
            self.EPSC[:, it] = d_EPSC_aux + EPSC
            self.d_EPSC[:, it] = d_EPSC_aux

    def calcium_steady_state(self, freq):
        """
        Approximation of steady-state response of calcium concentration
        :param freq:
        """
        tau_C = self.params['tao_Cai']
        # self.Cai_ss = freq * self.params['KCa'] + self.params['Cai_0']
        self.Cai_ss = (self.params['KCa'] / (tau_C * (1 - np.exp(-1 / (freq * tau_C))))) + self.params['Cai_0']

    def release_probability_steady_state(self, freq):
        """
        Approximation of steady-state response of release probability
        :param freq:
        """
        self.calcium_steady_state(freq)
        Cai_aux = np.power(self.Cai_ss, self.params['nHill'])
        Krel_aux = np.power(self.params['Krel_half'], self.params['nHill'])
        self.Prel_ss = (self.params['Prel_max'] * Cai_aux) / (Cai_aux + Krel_aux)

    def recovery_constant_steady_state(self, freq):
        """
        Approximation of steady-state response of recovery rate constant
        :param freq:
        """
        self.calcium_steady_state(freq)
        Krecov_0 = self.params['Krecov_0']
        Krecov_max = self.params['Krecov_max']
        Krecov_half = self.params['Krecov_half']
        Cai_ss = self.Cai_ss

        self.Krecov_ss = Krecov_0 + (Cai_ss * (Krecov_max - Krecov_0)) / (Cai_ss + Krecov_half)

    def releasable_vesicles_steady_state(self, freq):
        """
        Approximation of steady-state response of ready-to-release vesicles
        :param freq:
        """
        self.recovery_constant_steady_state(freq)
        self.release_probability_steady_state(freq)

        self.Rrel_ss = self.Krecov_ss / (self.Krecov_ss + (self.Prel_ss * freq))

    def resonance_freq(self):
        Cai_0 = self.params['Cai_0']
        KCa = self.params['KCa']
        Krel = self.params['Krel']
        Krel2 = self.params['Krel2']
        Prel_max = self.params['Prel_max']
        nHill = self.params['nHill']
        Krecov_half = self.params['Krel_half']
        Krecov_0 = self.params['Krecov_0'] * Krel2
        Krecov_aux = np.power(Krecov_half, nHill)
        Kca_aux = np.power(KCa, nHill) * Krel

        self.Fres = (-Cai_0 / KCa) + np.power((nHill * Krecov_aux * Krecov_0) / (Kca_aux * Prel_max), (1 / (1 + nHill)))

    def resonance_freq2(self):
        Cai_0 = self.params['Cai_0']
        KCa = self.params['KCa']
        Krel = self.params['Krel']
        Krel2 = self.params['Krel2']
        Prel_max = self.params['Prel_max']
        nHill = self.params['nHill']
        Krecov_half = self.params['Krel_half']
        Krecov_0 = self.params['Krecov_0'] * Krel2
        Krecov_aux = np.power(Krecov_half, nHill)
        Kca_aux = np.power(KCa, nHill) * Krel

        self.Fres2 = (1 / KCa) * (-Cai_0 + np.power(Prel_max * Krel / (KCa * nHill * Krecov_aux * Krecov_0),
                                                    (-1 / (nHill + 1))))

    def append_spike_event(self, t, output):
        """
        Storing spike events for each state variable given a t-time
        Parameters
        ----------
        t
        output
        """
        self.Cai_spike_events.append(self.Cai[:, t])
        self.Rrel_spike_events.append(self.Rrel[:, t])
        self.FluxGlu_spike_events.append(self.FluxGlu[:, t])
        self.Prel_spike_events.append(self.Prel[:, t])
        self.Krecov_spike_events.append(self.Krecov[:, t])
        self.time_spike_events.append(t)

        # Computing maximum EPSP response between the last and the current spike
        # If this is the first spike event
        if len(self.time_spike_events) > 1:
            spike_range = (self.time_spike_events[-2], self.time_spike_events[-1])
            self.compute_output_spike_event(spike_range, output)