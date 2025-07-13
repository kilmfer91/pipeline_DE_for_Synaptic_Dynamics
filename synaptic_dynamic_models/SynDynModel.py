import numpy as np


class SynDynModel:
    def __init__(self, n_syn=1):

        # time simulation variables
        self.dt = None
        self.time_vector = None
        self.L = None

        # input vector
        self.Input = None

        # Model variables - IMPLEMENT ON EACH CHILD CLASS

        # Spike events
        self.edge_detection = False
        self.output_spike_events = None
        self.time_spike_events = None
        self.output_steady_state = None
        self.efficacy = None
        self.efficacy_2 = None
        self.efficacy_3 = None
        self.t_steady_state = None
        self.t_max = None

        # derivative variables - IMPLEMENT ON EACH CHILD CLASS

        # Number of synapses
        self.n_syn = n_syn

        # Parameters - IMPLEMENT ON EACH CHILD CLASS
        self.params = None

        # Simulation parameters (default: 1 sec, with sampling frequency of 1KHz)
        self.sim_params = {'sfreq': 1000, 'max_t': 1.0, 'L': 1000, 'time_vector': np.arange(0, 1, 0.001)}

    def set_simulation_params(self, sim_params=None):
        # assert isinstance(self.sim_params, dict), "'sim_params' must be a dictionary"
        if sim_params is not None:
            assert isinstance(sim_params, dict), 'params should be a dict'
            for key, value in sim_params.items():
                if key in self.sim_params.keys():
                    self.sim_params[key] = value

        # time simulation variables
        self.dt = 1 / self.sim_params['sfreq']
        self.time_vector = np.arange(0, self.sim_params['max_t'], self.dt)
        self.L = int(self.sim_params['sfreq'] * self.sim_params['max_t'])

        # Coming from MSSM - Check if it is necesary
        if 'L' in self.sim_params.keys():
            self.L = self.sim_params['L']
        else:
            self.L = int(self.sim_params['sfreq'] * self.sim_params['max_t'])

        self.set_initial_conditions()

    def set_initial_conditions(self, Input=None):
        pass

    def set_model_params(self, model_params):
        pass

    def evaluate_model_euler(self, I_it, it):
        pass

    def evaluate_odeint(self, t, tspan, Input):
        pass

    def get_output(self):
        pass

    def detect_spike_event(self, t, output):
        """
        Parameters
        ----------
        t
        output
        """
        # Detecting raising edges
        # When t is 0
        if t == 0:
            # Detecting raising edges
            self.edge_detection = np.where(self.Input[:, t] > 0.0)[0]
        else:
            # Edge detector
            self.edge_detection = self.Input[:, t] > self.Input[:, t - 1]

        if np.sum(self.edge_detection) > 0:
            self.append_spike_event(t, output)

    def append_spike_event(self, t, output):
        """
        Storing spike events for each state variable given a t-time
        Parameters
        ----------
        t
        output
        """
        pass

    def compute_output_spike_event(self, spike_range, output):
        # print("TM, append_spike_event(), spike range ", spike_range, " in time ", spike_range[0])
        assert isinstance(spike_range, tuple), "Param 'spike_range' must be a tuple"
        assert len(spike_range) == 2, "Param 'spike_range' must be a tuple of 2 values"
        assert isinstance(spike_range[0], int), "first element of param 'spike_range' must be integer"
        assert isinstance(spike_range[1], int), "second element of param 'spike_range' must be integer"
        assert spike_range[1] >= spike_range[0], "Param 'spike_range' must contain order elements"
        assert isinstance(output, np.ndarray), "Param 'output' must be a numpy array"
        assert len(output.shape) == 2, "Param 'output' must be a 2D-array"
        assert output.shape[1] >= spike_range[1], ("second element of param 'spike_range' must be less or equal than "
                                                   "the length of param 'output'")
        if spike_range[1] == spike_range[0]:
            self.output_spike_events.append(output[:, spike_range[0]])
        else:
            # EPSP
            if np.sum(output) > 0:
                self.output_spike_events.append(np.max(output[:, spike_range[0]: spike_range[1]], axis=1))
            # EPSC
            else:
                self.output_spike_events.append(np.min(output[:, spike_range[0]: spike_range[1]], axis=1))

    @staticmethod
    def reach_steady_state(input_signals, size_window=10, epsilon=None):
        """
        check if 'input_signals' reaches the steady-state value, given an 'epsilon' for a window of size 'size_window'
        Parameters
        ----------
        input_signals: (numpy array) array of shape [n, m] for m-signals and n-time-steps
        size_window: (integer) size of the window
        epsilon: (array) array of shape [m, ] for m-signals
        Returns
        -------
        t_step: (integer) time step where the steady state of signals is reached
        """
        assert size_window + 1 < input_signals.shape[0], "window size must be smaller than signals duration"

        # number of signals
        num_signals = input_signals.shape[1]
        # getting maximum after computing abs(input_signals)
        maxi = np.max(np.abs(input_signals), axis=0)
        # Setting the epsilon error. By default, it is proportional to the max value of each signal
        if epsilon is None:
            epsilon = 1e-10 * maxi

        # Looping through all signals
        t_step = size_window + 1
        while t_step < input_signals.shape[0]:
            # Check for each signal if the epsilon condition is reached. Based on the difference between the maximum
            # and the minimum of each signal in the window.
            conditions = (np.max(input_signals[t_step - size_window:t_step, :], axis=0) -
                          np.min(input_signals[t_step - size_window:t_step, :], axis=0) < epsilon)

            # If all signals reach the steady-state value, then stop the loop
            if np.sum(conditions) == num_signals:
                break

            # Increment time-step
            t_step += 1

        return epsilon, t_step

    def get_efficacy_time_st(self):
        """

        Parameters
        ----------
        Returns
        -------

        """
        num_syn = self.n_syn
        output_spike_events = self.output_spike_events
        model_output = self.get_output()

        # array of output spike events
        output_aux = np.array(output_spike_events)
        # getting maximum after computing abs(output_aux)
        maxi = np.max(np.abs(output_aux), axis=0)
        # Setting the epsilon error
        epsilon = 1e-10 * maxi
        # Initializing time to reach steady-state
        t_ss = np.zeros(num_syn, dtype=int)  # IMPORTANT, initialize with zeros
        # Matrix of conditions, to check if synapses reach the steady-state value (last 10 spike events less than error)
        conditions_matrix = np.zeros((num_syn, len(output_aux)), dtype=bool)
        evaluated_syn = np.zeros(num_syn, dtype=int)
        end_loop_break = False

        # Looping through the output spike events
        for i in range(1, len(output_aux)):
            # when there is at least 10 events
            if i >= 10:
                # Check for each synapse if the epsilon condition is reached
                conditions = np.max(output_aux[i - 10:i, :], axis=0) - np.min(output_aux[i - 10:i, :], axis=0) < epsilon

                # Update matrix of conditions
                conditions_matrix[:, i] = conditions

                # If all synapses reach the steady-state value, then stop the loop
                if np.sum(conditions) == num_syn:
                    # Update with True for all last positions in matrix of conditions
                    conditions_matrix[:, i: len(output_aux)] = True
                    end_loop_break = True
                    break

        # If the loop does not break, update with True all last positions of matrix of conditions
        if not end_loop_break:
            conditions_matrix[:, -1] = True

        # Find unique values, to compute the sample when each synase reach the steady state value
        unique_cond = np.unique(conditions_matrix, return_index=True, axis=1)

        if num_syn == unique_cond[1][1:].shape[0]:
            t_ss = unique_cond[1][1:]
        else:
            max_step_cond = np.max(unique_cond[1])
            for j in range(1, len(unique_cond[1])):
                a = np.logical_xor(unique_cond[0][:, j - 1], unique_cond[0][:, j])
                # evaluated_syn = np.logical_not(np.logical_or(evaluated_syn, a))
                b = np.logical_and(a, np.logical_not(evaluated_syn)) * unique_cond[1][j]
                aux_t = t_ss + b

                # checking if the computation of step steady-state is wrong
                if np.max(aux_t) > max_step_cond:
                    ind_wrong_tss = np.argmax(t_ss + b)
                    aux_t[ind_wrong_tss] -= b[ind_wrong_tss]
                t_ss = aux_t

                evaluated_syn = np.logical_or(evaluated_syn, a)

        efficacy = np.abs((output_aux[t_ss] / output_aux[0, :])[0])  # np.abs(output_aux[t_ss][0])  #
        output_steady_state = output_aux[t_ss][0]
        t_steady_state = t_ss

        # If the maximum response is similar to the last-input-spike response, then set the index to
        # the length of the input
        ss_output_aux = np.array(output_spike_events)
        abs_output = np.abs(model_output)
        ind_max_out = abs_output.argmax(axis=1)
        out_last_spike = output_aux[-1]

        conditions = abs_output[:, ind_max_out].diagonal() - output_steady_state < 1e-9
        aux_time_max = np.logical_not(conditions) * ind_max_out + conditions * model_output.shape[1]
        aux_eff_2 = output_steady_state / model_output[:, ind_max_out].diagonal()
        aux_eff_3 = np.abs(model_output[:, ind_max_out].diagonal() / ss_output_aux[0, :])

        return efficacy, aux_eff_2, aux_eff_3, output_steady_state, t_steady_state, aux_time_max

    @staticmethod
    def get_efficacy_time_ss(num_syn, output_spike_events):
        """

        Parameters
        ----------
        num_syn
        output_spike_events

        Returns
        -------

        """
        # array of output spike events
        output_aux = np.array(output_spike_events)
        # getting maximum after computing abs(output_aux)
        maxi = np.max(np.abs(output_aux), axis=0)
        # Setting the epsilon error
        epsilon = 1e-10 * maxi
        # Initializing time to reach steady-state
        t_ss = np.zeros(num_syn, dtype=int)  # IMPORTANT, initialize with zeros
        # Matrix of conditions, to check if synapses reach the steady-state value (last 10 spike events less than error)
        conditions_matrix = np.zeros((num_syn, len(output_aux)), dtype=bool)
        evaluated_syn = np.zeros(num_syn, dtype=int)
        end_loop_break = False

        # Looping through the output spike events
        for i in range(1, len(output_aux)):
            # when there is at least 10 events
            if i >= 10:
                # Check for each synapse if the epsilon condition is reached
                conditions = np.max(output_aux[i - 10:i, :], axis=0) - np.min(output_aux[i - 10:i, :], axis=0) < epsilon

                # Update matrix of conditions
                conditions_matrix[:, i] = conditions

                # If all synapses reach the steady-state value, then stop the loop
                if np.sum(conditions) == num_syn:
                    # Update with True for all last positions in matrix of conditions
                    conditions_matrix[:, i: len(output_aux)] = True
                    end_loop_break = True
                    break

        # If the loop does not break, update with True all last positions of matrix of conditions
        if not end_loop_break:
            conditions_matrix[:, -1] = True

        # Find unique values, to compute the sample when each synase reach the steady state value
        unique_cond = np.unique(conditions_matrix, return_index=True, axis=1)

        if num_syn == unique_cond[1][1:].shape[0]:
            t_ss = unique_cond[1][1:]
        else:
            max_step_cond = np.max(unique_cond[1])
            for j in range(1, len(unique_cond[1])):
                a = np.logical_xor(unique_cond[0][:, j - 1], unique_cond[0][:, j])
                b = np.logical_and(a, np.logical_not(evaluated_syn)) * unique_cond[1][j]
                aux_t = t_ss + b

                # checking if the computation of step steady-state is wrong
                if np.max(aux_t) > max_step_cond:
                    ind_wrong_tss = np.argmax(t_ss + b)
                    aux_t[ind_wrong_tss] -= b[ind_wrong_tss]
                t_ss = aux_t

                evaluated_syn = np.logical_or(evaluated_syn, a)

        efficacy = output_aux[t_ss][0]  # (output_aux[t_ss] / output_aux[0, :])[0]  #
        output_steady_state = output_aux[t_ss][0]
        t_steady_state = t_ss

        return efficacy, output_steady_state, t_steady_state

    # @staticmethod
    def run_model(self, time_vector, *args, **kwargs):
        """
        Simulation of synapses of SynDynModel objects for a given input (kwargs['Input']), a given set of parameters
        (names of parameters - kwargs['params_name'], parameters - args), and the way to compute the time functions
        (kwargs['ODE_model']). The output can be the time-series of the SD_model's postsynaptic response, or it can be
        only the responses to the input spikes (kwargs['only spikes']).
        :param time_vector: (numpy array [t, ]) time vector for simulation.
        :param args: (numpy array [Num_P, ]) set of Num_P parameters for the SD model to simulate.
        :param kwargs: (dictionary {'Input': a, 'params_name': b, 'mode': c, 'only spikes': d})
                       Extra parameters to specify:
                       a (numpy array [t, ]) the input spike train.
                       b (list) the name of parameters. It must have the same size as *args, and it must  contain at
                         least one name of the parameteres of the SD model:
                       c (String) the type of ODE solver supported by SynDynModel. Up to now, there are two options:
                         'ODE': for Euler method.
                         'Analytical': for analytical solution of equations (no ODE solver).
                       d (boolean) if True, the output of the model is only the postsynaptic response of the input
                         spikes, otherwise it is the time-series of the postsynaptic response.
        :return: model_SD.get_output() (numpy array [t, ]) postsynaptic response.

        As example, the parameters of the MSSM model should look like this:
        param = {kwargs['params_name'][0]: args[0], kwargs['params_name'][1]: args[1],
                 kwargs['params_name'][2]: args[2], kwargs['params_name'][3]: args[3],
                 kwargs['params_name'][4]: args[4], kwargs['params_name'][5]: args[5],
                 kwargs['params_name'][6]: args[6], kwargs['params_name'][7]: args[7],
                 kwargs['params_name'][8]: args[8], kwargs['params_name'][9]: args[9]}
        given the numpy array of args = np.array([args[0], args[1], args[2], args[3], args[4],
                                                  args[5], args[6], args[7], args[8], args[9]])
        and the list of kwargs['params_name'] = ['tau_c', 'alpha', 'V0', 'tau_v', 'P0',
                                                 'k_NtV', 'k_Nt', 'tau_Nt', 'k_EPSP', 'tau_EPSP']
        """

        # Local variables
        params_name = kwargs['params_name']
        Input = kwargs['Input']
        output_model = None
        # model_SD = kwargs['model']
        param = {}
        return_only_spike_event = False
        output_spike_events = []
        L = len(time_vector)

        # Creating param dictionary
        for i in range(len(args)):
            param[params_name[i]] = args[i]

        # Update parameters and initial conditions
        self.set_model_params(param)
        self.set_initial_conditions()

        # If there is a neuron model
        model_neuron = None
        if "model_neuron" in kwargs:
            if kwargs['model_neuron'] is not None:
                model_neuron = kwargs['model_neuron']
                model_neuron.set_simulation_params()

        # Returning the entire time series or only the response after an input spike
        if "only spikes" in kwargs:
            return_only_spike_event = kwargs['only spikes']

        # Evaluating SD model
        if kwargs['mode'] == "odeint":
            self.evaluate_odeint_time(time_vector, Input)
        else:
            for t in range(L):
                if kwargs['mode'] == "ODE":
                    self.evaluate_model_euler(Input[t], t)
                else:
                    assert False, "'ODE mode' must be either 'ODE' or 'odeint'"

                # Evaluating neuron response
                if model_neuron is not None:
                    model_neuron.update_state(self.get_output[:, it], t)
                    # Output based on the neuron model
                    output_model = model_neuron.membrane_potential
                else:
                    # Output based on the SD model
                    output_model = self.get_output()

                # Detecting spike events and storing model output
                self.detect_spike_event(t, output_model)

            # Computing output spike event in the last ISI
            t = L
            spike_range = (self.time_spike_events[-1], t)
            self.compute_output_spike_event(spike_range, output_model)

        if return_only_spike_event:
            return np.array(self.output_spike_events).T
        if model_neuron is not None:
            return model_neuron.membrane_potential
        else:
            return self.get_output()
