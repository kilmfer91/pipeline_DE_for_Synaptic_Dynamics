from utils import *
from scipy.optimize import curve_fit, differential_evolution, NonlinearConstraint

from synaptic_dynamic_models.MSSM import MSSM_model
from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.SynDynModel import SynDynModel
from libraries.frequency_analysis import Freq_analysis


class Fit_params_SD:

    def __init__(self, input_signal, reference_signal, model_SD, dict_params, DE_params):

        # Input
        self.output_factor = None
        self.Input, self.reference_signal = None, None

        # model vars
        self.params_name = None
        self.bo = None
        self.ini = None
        self.parameters = None
        self.kwargs_model = None
        self.model_str = None
        self.model_SD = None
        self.fitted_parameters = None
        self.results_optimization = None
        self.types_ODE = ['ODE', 'odeint']
        self.types_optimizer = ['NLS', 'DE']
        self.fit_new_params = None
        self.dict_params = None
        self.DE_params = None
        self.fa = None

        # Time vars
        self.sfreq, self.dt, self.time_vector, self.L, self.max_t = None, None, None, None, None

        # Initialization
        self.set_dict_params(dict_params)
        self.model_str = dict_params['model_str']
        self.set_model(model_SD)
        self.set_time_vars(model_SD.sim_params)
        if not self.dict_params['only_spikes'] and not self.dict_params['frequency_reference']:
            assert len(reference_signal) == len(input_signal), "Input and reference must have the same shape"
        self.set_input(input_signal)
        self.set_reference(reference_signal)
        self.set_DE_params(DE_params)

    def set_time_vars(self, sim_params):
        """
        Parameters
        ----------
        sim_params
        Returns
        -------
        """
        # parameters
        self.sfreq = sim_params['sfreq']
        self.dt = 1. / self.sfreq
        self.time_vector = sim_params['time_vector']
        self.L = sim_params['L']
        self.max_t = sim_params['max_t']

    def set_model(self, model_SD):
        """

        Parameters
        ----------
        Returns
        -------
        """
        # """
        assert isinstance(model_SD, SynDynModel), "'SD model' must be an instance of SynDynModel"
        self.model_SD = model_SD
        # """

    def set_dict_params(self, dict_params):
        self.assert_dict_params(dict_params)
        self.dict_params = dict_params
        self.bo = self.dict_params['bo']
        self.params_name = self.dict_params['params_name']
        self.kwargs_model = {'Input': self.Input, 'params_name': self.params_name, 'mode': dict_params['ODE_mode'],
                             'only spikes': dict_params['only_spikes']}

    def set_input(self, input_signal):
        self.Input = input_signal

    def set_reference(self, ref_signal):
        self.reference_signal = ref_signal

    def set_DE_params(self, params):
        self.DE_params = params

    def fun_aux_de(self, x):
        eval_model = self.model_SD.run_model(self.time_vector, *x, **self.kwargs_model)
        # print("model evaluation: ", rmse(ref_vector, eval_model), self.num_DE_executed)
        return rmse(self.reference_signal * self.output_factor, eval_model)

    def fun_aux_freq_de(self, x):
        # Creating param dictionary to run freq analysis on the new parameters x
        param = {}
        for i in range(len(x)):
            param[self.kwargs_model['params_name'][i]] = x[i]
        self.model_SD.set_model_params(param)

        # Creating instance of Freq_analysis
        self.fa = Freq_analysis(sim_params=self.model_SD.sim_params, loop_f=self.dict_params['ref_freq_vector'],
                                n_syn=x.shape[1])
        self.fa.set_model(model_str=self.model_SD, name_params=list(self.model_SD.params.keys()),
                          model_params=list(self.model_SD.params.values()))

        # Run frequency analysis
        self.fa.run()

        return rmse(self.reference_signal * self.output_factor, self.fa.efficacy)

    def run_DE(self, bo_de, freq_ref=False):
        fun_aux = None
        if freq_ref:
            fun_aux = self.fun_aux_freq_de
        else:
            fun_aux = self.fun_aux_de
        return differential_evolution(fun_aux, bounds=bo_de, strategy=self.DE_params[0],
                                      maxiter=self.DE_params[1], popsize=self.DE_params[2], tol=self.DE_params[3],
                                      mutation=self.DE_params[4], recombination=self.DE_params[5],
                                      seed=self.DE_params[6], callback=self.DE_params[7], disp=self.DE_params[8],
                                      polish=self.DE_params[9], init=self.DE_params[10], atol=self.DE_params[11],
                                      updating=self.DE_params[12], workers=self.DE_params[13],
                                      constraints=self.DE_params[14], x0=self.DE_params[15],
                                      integrality=self.DE_params[16], vectorized=self.DE_params[17])

    def run_fit(self):
        # Getting other parameters
        path_to_vars = self.dict_params['path']
        ODE_mode = self.dict_params['ODE_mode']
        ind_experiment = self.dict_params['ind_experiment']
        only_spikes = self.dict_params['only_spikes']
        suf_file = self.dict_params['description_file']
        freq_ref = self.dict_params['frequency_reference']
        no_ini_conditions = False
        self.output_factor = 1.
        if 'no_initial_cond_mssm' in self.dict_params.keys():
            no_ini_conditions = self.dict_params['no_initial_cond_mssm']
        if 'output_factor' in self.dict_params.keys():
            self.output_factor = self.dict_params['output_factor']

        # Escalation of the ref_vector given the output factor
        ref_vector = self.reference_signal * self.output_factor

        # Creating name of new file
        sufix = "_" + str(ind_experiment)
        aux_file = path_to_vars + "param_estimation_" + self.model_str + "_" + suf_file + sufix + ".pkl"

        # If the auxiliar file exists, then fit_new_params is set to False and retrieve the data from the file.
        self.fit_new_params = True
        if os.path.isfile(aux_file):
            self.fit_new_params = False

        # If fit_new_params, new params should be computed and stored
        if self.fit_new_params:
            print("Evaluating curve fitting for model", self.model_str)

            # defining kwargs
            self.kwargs_model = {'Input': self.Input, 'params_name': self.params_name, 'mode': ODE_mode,
                                 'only spikes': only_spikes}

            # Differential Evolution
            # Formatting boundaries
            bo_de = []
            for i in range(len(self.bo[0])):
                bo_de.append((self.bo[0][i], self.bo[1][i]))
            # Running DE
            results_de = self.run_DE(bo_de, freq_ref)  # differential_evolution(fun_aux_de, bounds=bo_de, disp=True)
            # Saving results
            parametersoln = results_de.x
            self.results_optimization = results_de

            # Varibles to save
            dict_to_save = {"params_name": self.params_name, "bounds": self.bo, "params": parametersoln,
                            "reference": ref_vector, "scaling_factor_ref": self.output_factor,
                            "res_optimization": self.results_optimization}

            while check_file(aux_file):
                ind_experiment += 1
                sufix = "_" + str(ind_experiment)
                aux_file = (path_to_vars + "param_estimation_" + self.model_str + "_" + suf_file + sufix + ".pkl")

            # Save the results as a .pkl file in the folder "variables"
            saved_file = open(aux_file, 'wb')
            pickle.dump(dict_to_save, saved_file)
            saved_file.close()

            print("For experiment ", ind_experiment, ", parameters are: ", parametersoln)
            self.fitted_parameters = parametersoln

        # Otherwise try to upload parameters
        else:
            assert os.path.isfile(aux_file), "File %s does not exist" % aux_file
            # If the file is already saved, then load it
            read_file = open(aux_file, 'rb')
            dict_to_read = pickle.load(read_file)

            self.results_optimization = dict_to_read['res_optimization']
            self.fitted_parameters = dict_to_read['res_optimization']['population']  # dict_to_read['params']
            self.bo = dict_to_read['bounds']
            ref_vector = dict_to_read['reference']
            output_factor = dict_to_read['scaling_factor_ref']
            self.kwargs_model = {'Input': self.Input, 'params_name': dict_to_read['params_name'], 'mode': ODE_mode,
                                 'only spikes': only_spikes}
            # ref_vector *= output_factor

        return self.fitted_parameters

    def evaluate_model(self, params):
        eval_model = self.model_SD.run_model(self.time_vector, *params, **self.kwargs_model)

    def assert_dict_params(self, dict_params):
        assert isinstance(dict_params, dict), "dict_params must be a Dictionary"
        assert 'model_str' in dict_params.keys(), "dict_params must have 'model_str' as key"
        assert 'params_name' in dict_params.keys(), "dict_params must have 'params_name' as key"
        assert 'bo' in dict_params.keys(), "dict_params must have 'bo' as key"
        assert 'ODE_mode' in dict_params.keys(), "dict_params must have 'ODE_mode' as key"
        assert 'ind_experiment' in dict_params.keys(), "dict_params must have 'ind_experiment' as key"
        assert 'only_spikes' in dict_params.keys(), "dict_params must have 'only_spikes' as key"
        assert 'path' in dict_params.keys(), "dict_params must have 'path' as key"
        assert 'description_file' in dict_params.keys(), "dict_params must have 'description_file' as key"
        assert 'output_factor' in dict_params.keys(), "dict_params must have 'output_factor' as key"
        assert isinstance(dict_params['model_str'], str), "'model' must be a string"
        assert isinstance(dict_params['params_name'], list), "'params_name' must be a list"
        assert isinstance(dict_params['bo'], tuple), "'bo' must be a tuple"
        assert len(dict_params['bo']) == 2, "'bo' must be a tuple of 2 values"
        assert isinstance(dict_params['bo'][0], tuple), "'bo' in 1st position must be a tuple"
        assert isinstance(dict_params['bo'][1], tuple), "'bo' in 2nd position must be a tuple"
        aux_cond = len(dict_params['bo'][0]) == len(dict_params['params_name']) and \
                   len(dict_params['bo'][1]) == len(dict_params['params_name'])
        assert aux_cond, "length of the 2 tuples of 'bo' must be equal to length of 'params_name'"
        assert isinstance(dict_params['ODE_mode'], str), "dict_params['ODE_mode'] must be a string"
        assert dict_params['ODE_mode'] in self.types_ODE, "dict_params['ODE_mode'] must be in " + str(self.types_ODE)
        assert isinstance(dict_params['ind_experiment'], int), "dict_params['ind_experiment'] must be an int"
        assert dict_params['ind_experiment'] >= 0, "dict_params['ind_experiment'] must be positive or zero"
        assert isinstance(dict_params['only_spikes'], bool), "'only_spikes' must be a boolean"
        assert os.path.isdir(dict_params['path']), "Folder %s does not exist" % dict_params['path']
        assert isinstance(dict_params['description_file'], str), "dict_params['description_file'] must be a string"
        assert isinstance(dict_params['output_factor'], float), "dict_params['output_factor'] must be a float"
        assert 'frequency_reference' in dict_params.keys(), "dict_params must have 'frequency_reference' as key"
        assert isinstance(dict_params['frequency_reference'], bool), "dict_params['frequency_reference'] must be bool"
        if dict_params['frequency_reference']:
            assert 'ref_freq_vector' in dict_params.keys(), "dict_params must have 'ref_freq_vector' as key"
            assert isinstance(dict_params['ref_freq_vector'], list), "'ref_freq_vector' must be a list"
