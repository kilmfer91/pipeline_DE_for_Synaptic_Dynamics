from utils import *
from synaptic_dynamic_models.MSSM import MSSM_model
from synaptic_dynamic_models.LAP import LAP_model
from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.SynDynModel import SynDynModel


class Freq_analysis:
    def __init__(self, sim_params, loop_f=None, stoch_input=False, seed=None, input_factor=1.0, n_syn=1):
        """
        Parameters
        ----------
        sim_params
        loop_f
        stoch_input
        seed
        input_factor
        n_syn
        """
        self.aux_response_t = None
        self.Input_vector = None
        self.ss_output = None
        self.ss_t = None
        self.model_stp = None
        self.name_params = None
        self.pa = None
        self.model_str = None

        # Number of synapses
        self.n_syn = n_syn
        # Time vars
        self.sfreq = sim_params['sfreq']
        self.dt = None
        self.time_vector = None
        self.L = None
        self.max_t = None
        self.end_t = None
        self.max_t_plot = None

        # Output
        self.time_max = None
        self.time_ss = None
        self.efficacy = None
        self.efficacy_2 = None
        self.efficacy_3 = None

        # Output - trials
        self.fa_eff1 = []
        self.fa_eff2 = []
        self.fa_eff3 = []
        self.fa_t_ss = []
        self.fa_t_max = []
        self.MSSM_syn_with_weird_freq = [[], []]

        self.loop_frequencies = [i for i in range(1, 101)]
        if loop_f is not None:
            self.loop_frequencies = loop_f

        # Default settings of time and input-vector
        self.sim_params = sim_params
        self.set_time_vars(sim_params)
        self.reset_ini_cond(stoch_input, seed, input_factor)

        # Model of neuron if necessary
        self.model_neuron = None

    def set_time_vars(self, sim_params):
        """

        Parameters
        ----------
        sim_params
        -------
        """

        # parameters
        self.dt = 1. / self.sfreq
        self.time_vector = np.arange(0, sim_params['max_t'], self.dt)
        self.L = sim_params['L']
        self.max_t = sim_params['max_t']
        self.end_t = self.max_t - 0.1  # end_t

    def set_model(self, model_str, name_params, model_params, sim_params=None, model_neuron=None):
        """

        Parameters
        ----------
        model_str
        name_params
        model_params
        sim_params: (None)
        model_neuron: (None)
        -------

        """
        self.model_str = model_str

        # CREATING MODELS
        if self.model_str == "MSSM":
            self.model_stp = MSSM_model(n_syn=self.n_syn)
        if self.model_str == "LAP":
            self.model_stp = LAP_model(n_syn=self.n_syn)
        if self.model_str == "TM":
            self.model_stp = TM_model(n_syn=self.n_syn)

        if isinstance(model_str, SynDynModel):
            self.model_stp = model_str

        if model_neuron is not None:
            self.model_neuron = model_neuron

        if sim_params is None:
            sim_params = self.sim_params

        self.model_stp.set_simulation_params(sim_params)
        self.pa = model_params
        self.name_params = name_params

    def reset_ini_cond(self, stochastic=False, seed=None, input_factor=1.0):
        """
        Returns
        -------
        """
        self.set_input_vector(stochastic, seed, input_factor)
        self.ss_output = np.empty(self.L)
        self.ss_t = np.empty(self.L)
        self.time_max = []
        self.time_ss = []
        self.efficacy = []
        self.efficacy_2 = []
        self.efficacy_3 = []

    def set_input_vector(self, stochastic=False, seed=None, input_factor=1.0):
        """
        Parameters
        ----------
        stochastic
        seed
        input_factor
        Returns
        -------
        """
        num_freq = len(self.loop_frequencies)
        L = self.L
        self.Input_vector = np.zeros((num_freq, L))
        i = 0
        for r in self.loop_frequencies:
            if not stochastic:
                i_ca = input_spike_train(self.sfreq, r, self.end_t)
                i_ca = np.concatenate((i_ca, np.zeros(L - i_ca.shape[0])))
                # ICa[0] = 0
                self.Input_vector[i, :] = np.roll(i_ca, 1) * input_factor
            else:
                self.Input_vector[i, :] = poisson_generator(self.dt, self.time_vector, r, 1, seed) * input_factor
                self.Input_vector[i, 0] = 1
            i += 1

    def run(self, stochastic=False, num_trials=10, f_int=None):
        """

        Parameters
        ----------
        stochastic
        num_trials
        f_int
        Returns
        -------

        """
        if stochastic:
            for i in range(num_trials):
                seed = int(su([1, 10e4]))
                print("freq analaysis No. ", i, "seed ", seed)
                self.reset_ini_cond(stochastic=stochastic, seed=seed)
                self.run_analysis()
                self.fa_eff1.append(self.efficacy)
                self.fa_eff2.append(self.efficacy_2)
                self.fa_eff3.append(self.efficacy_3)
                self.fa_t_ss.append(self.time_ss)
                self.fa_t_max.append(self.time_max)

            self.fa_eff1 = np.array(self.fa_eff1)
            self.fa_eff2 = np.array(self.fa_eff2)
            self.fa_eff3 = np.array(self.fa_eff3)
            self.fa_t_ss = np.array(self.fa_t_ss)
            self.fa_t_max = np.array(self.fa_t_max)
        else:
            self.run_analysis(freq_interest=f_int)

    def run_analysis(self, freq_interest=None, stochastic=False, num_trials=10):
        """
        freq_interest
        stochastic
        num_trials
        Returns
        -------
        """
        # Creating local variable for time_vector and end_t (in case of reducing it during freq analysis)
        time_vector = self.time_vector
        end_t = self.end_t

        # Looping through all frequencies
        i = 0
        num_freq = len(self.loop_frequencies)
        while i < num_freq:  # 59:
            r = self.loop_frequencies[i]
            Input = self.Input_vector[i, :]

            # Aux ini
            self.ss_output[:] = np.nan
            self.ss_t[:] = np.nan

            # Additional arguments
            kwargs_model = {'model': self.model_stp, 'Input': Input, 'params_name': self.name_params, 'mode': 'ODE',
                            'model_neuron': self.model_neuron}

            # Evaluating the model
            self.model_stp.run_model(time_vector, *self.pa, **kwargs_model)

            # Steady-state calculations
            ss_output_aux = np.array(self.model_stp.output_spike_events)

            # Computing value and index of output's maximum
            model_output = None

            if self.model_neuron is not None:
                model_output = self.model_neuron.membrane_potential
            else:
                model_output = self.model_stp.get_output()
            abs_output = np.abs(model_output)
            ind_max_out = abs_output.argmax(axis=1)

            # Time to reach steady-state values
            aux_eff, aux_eff_2, aux_eff_3, output_st, t_st, aux_t_max = self.model_stp.get_efficacy_time_st()
            aux_time_ss = t_st / r
            aux_time_max = aux_t_max * self.dt

            # Update times of maximum and stedy-state
            self.time_max.append(aux_time_max)
            self.time_ss.append(aux_time_ss)
            # Update synaptic efficacies
            self.efficacy.append(aux_eff)
            self.efficacy_2.append(aux_eff_2)
            self.efficacy_3.append(aux_eff_3)
            # """

            # """
            # Updating time_vector to time of steady state + 1 sec.
            aux_t_ss = np.max(self.time_ss[-1])
            if aux_t_ss == self.max_t and r > 1:
                time_vector = self.time_vector
                end_t = self.end_t
            elif aux_t_ss < (self.max_t - 1.0) and r > 1:
                # updating time vector and end_t
                time_vector = np.arange(0, aux_t_ss + 0.2, self.dt)
                end_t = aux_t_ss + 0.1

            """
            # if r % 10 == 0:
            print("Freq %.1f: Eff %.4f, Eff2 %.4f, Eff3 %.4f, t_st %.4f, t_max %.4f " % (r, np.mean(aux_eff),
                                                                                         np.mean(aux_eff_2[-1]),
                                                                                         np.mean(aux_eff_3[-1]),
                                                                                         np.mean(aux_time_ss[0]),
                                                                                         np.mean(aux_time_max[0])))
            # """
            i += 1

        self.time_max = np.array(self.time_max).T
        self.time_ss = np.array(self.time_ss).T
        self.efficacy = np.array(self.efficacy).T
        self.efficacy_2 = np.array(self.efficacy_2).T
        self.efficacy_3 = np.array(self.efficacy_3).T
