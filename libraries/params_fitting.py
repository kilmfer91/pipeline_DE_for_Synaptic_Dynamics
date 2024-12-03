from utils import *
from scipy.optimize import curve_fit, differential_evolution, NonlinearConstraint

from synaptic_dynamic_models.MSSM import MSSM_model
from synaptic_dynamic_models.TM import TM_model
from synaptic_dynamic_models.SynDynModel import SynDynModel


class Fit_params_SD:

    def __init__(self, input_signal, reference_signal, model_SD, dict_params, DE_params):

        # Input
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

        # Time vars
        self.sfreq, self.dt, self.time_vector, self.L, self.max_t = None, None, None, None, None

        # Initialization
        self.set_dict_params(dict_params)
        self.model_str = dict_params['model_str']
        self.set_model(model_SD)
        self.set_time_vars(model_SD.sim_params)
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

    def sample_ini_params(self, dict_params=None):
        self.params_name = dict_params['params_name']
        self.bo = dict_params['bo']
        self.ini = dict_params['ini']
        ini = self.ini
        self.parameters = []
        for i in range(len(dict_params['type_sample_param'])):
            k = dict_params['type_sample_param'][i]
            if k == 'su':
                self.parameters.append(su(ini[i]))
            if k == 'sn':
                self.parameters.append(sn(ini[i]))
        print("Fit_params_stp, sample_ini_params, ", self.parameters)
        return np.array(self.parameters, dtype='float64')

    def set_dict_params(self, dict_params):
        self.assert_dict_params(dict_params)
        self.dict_params = dict_params

    def set_input(self, input_signal):
        self.Input = input_signal

    def set_reference(self, ref_signal):
        self.reference_signal = ref_signal

    def set_DE_params(self, params):
        self.DE_params = params

    def fun_aux_de(self, x):
        # self.num_DE_executed += 1
        eval_model = self.model_SD.run_model(self.time_vector, *x, **self.kwargs_model)
        # print("model evaluation: ", rmse(ref_vector, eval_model), self.num_DE_executed)
        return rmse(self.reference_signal * self.output_factor, eval_model)

    def run_DE(self, bo_de):
        return differential_evolution(self.fun_aux_de, bounds=bo_de, strategy=self.DE_params[0], maxiter=self.DE_params[1],
                                      popsize=self.DE_params[2], tol=self.DE_params[3], mutation=self.DE_params[4],
                                      recombination=self.DE_params[5], seed=self.DE_params[6],
                                      callback=self.DE_params[7], disp=self.DE_params[8], polish=self.DE_params[9],
                                      init=self.DE_params[10], atol=self.DE_params[11], updating=self.DE_params[12],
                                      workers=self.DE_params[13], constraints=self.DE_params[14], x0=self.DE_params[15],
                                      integrality=self.DE_params[16], vectorized=self.DE_params[17])

    def run_fit(self):

        # self.num_DE_executed = 0

        # Checking input and reference
        assert self.Input is not None, "Input is not defined"
        # assert len(self.reference_signal) == len(self.Input), "Input and reference must have the same shape"

        # Checking content of dict_params
        # self.set_dict_params(dict_params)

        # Getting other parameters
        path_to_vars = self.dict_params['path']
        ODE_mode = self.dict_params['ODE_mode']
        optimizer_mode = self.dict_params['optimizer_mode']
        ind_experiment = self.dict_params['ind_experiment']
        only_spikes = self.dict_params['only_spikes']
        suf_file = self.dict_params['description_file']
        no_ini_conditions = False
        self.output_factor = 1.
        if 'no_initial_cond_mssm' in self.dict_params.keys():
            no_ini_conditions = self.dict_params['no_initial_cond_mssm']
        if 'output_factor' in self.dict_params.keys():
            self.output_factor = self.dict_params['output_factor']

        # Escalation of the ref_vector given the output factor
        ref_vector = self.reference_signal *  self.output_factor

        # Creating name of new file
        sufix = "_" + str(ind_experiment)
        aux_file = path_to_vars + "param_estimation_" + ODE_mode + "_" + optimizer_mode + "_" + self.model_str + "_" + \
                   suf_file + sufix + ".pkl"

        # If the auxiliar file exists, then fit_new_params is set to False and retrieve the data from the file.
        self.fit_new_params = True
        if os.path.isfile(aux_file):
            self.fit_new_params = False

        # If fit_new_params, new params should be computed and stored
        if self.fit_new_params:
            print("Evaluating curve fitting for model", self.model_str)

            # Sampling initial parameters
            ini_params = self.sample_ini_params(dict_params=self.dict_params)

            # defining kwargs
            self.kwargs_model = {'Input': self.Input, 'params_name': self.params_name, 'mode': ODE_mode,
                                 'only spikes': only_spikes}

            def fun_aux_nls(time_vector, *ini_params):
                eval_model = self.model_SD.run_model(time_vector, *ini_params, **self.kwargs_model)
                return eval_model

            # Non-linear least squares optimization method
            if optimizer_mode == 'NLS':
                parametersoln, pcov, curve_a, curve_b, curve_c = curve_fit(fun_aux_nls, self.time_vector, ref_vector,
                                                                           p0=ini_params, bounds=self.bo,
                                                                           full_output=True)
                self.results_optimization = {'x': parametersoln, 'pcov': pcov, 'curve_a': curve_a, 'curve_b': curve_b,
                                             'curve_c': curve_c}
            # Differential Evolution
            elif optimizer_mode == 'DE':
                # Formatting boundaries
                bo_de = []
                for i in range(len(self.bo[0])):
                    bo_de.append((self.bo[0][i], self.bo[1][i]))
                # constraint on tau Nt
                # nlc = NonlinearConstraint(fun_constraint_de, self.bo[0][7], self.bo[1][7])
                results_de = self.run_DE(bo_de)  # differential_evolution(fun_aux_de, bounds=bo_de, disp=True)
                parametersoln = results_de.x
                self.results_optimization = results_de
            else:
                assert False, "optimization method not recognized"

            # Varibles to save
            dict_to_save = {"params_name": self.params_name, "bounds": self.bo, "params": parametersoln,
                            "reference": ref_vector, "scaling_factor_ref": self.output_factor,
                            "res_optimization": self.results_optimization}

            while check_file(aux_file):
                ind_experiment += 1
                sufix = "_" + str(ind_experiment)
                aux_file = (path_to_vars + "param_estimation_" + ODE_mode + "_" + optimizer_mode + "_" +
                            self.model_str + "_" + suf_file + sufix + ".pkl")

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

            self.fitted_parameters = dict_to_read['params']
            self.results_optimization = dict_to_read['res_optimization']
            self.bo = dict_to_read['bounds']
            ref_vector = dict_to_read['reference']
            output_factor = dict_to_read['scaling_factor_ref']
            self.kwargs_model = {'Input': self.Input, 'params_name': dict_to_read['params_name'], 'mode': ODE_mode,
                                 'only spikes': only_spikes}
            # ref_vector *= output_factor

        return self.fitted_parameters

    def evaluate_model(self):
        eval_model = self.model_SD.run_model(self.time_vector, *self.fitted_parameters, **self.kwargs_model)

    def assert_dict_params(self, dict_params):
        assert isinstance(dict_params, dict), "dict_params must be a Dictionary"
        assert 'model_str' in dict_params.keys(), "dict_params must have 'model_str' as key"
        assert 'params_name' in dict_params.keys(), "dict_params must have 'params_name' as key"
        assert 'bo' in dict_params.keys(), "dict_params must have 'bo' as key"
        assert 'ini' in dict_params.keys(), "dict_params must have 'ini' as key"
        assert 'type_sample_param' in dict_params.keys(), "dict_params must have 'type_sample_param' as key"
        assert 'ODE_mode' in dict_params.keys(), "dict_params must have 'ODE_mode' as key"
        assert 'optimizer_mode' in dict_params.keys(), "dict_params must have 'optimizer_mode' as key"
        assert 'ind_experiment' in dict_params.keys(), "dict_params must have 'ind_experiment' as key"
        assert 'only_spikes' in dict_params.keys(), "dict_params must have 'only_spikes' as key"
        assert 'path' in dict_params.keys(), "dict_params must have 'path' as key"
        assert 'description_file' in dict_params.keys(), "dict_params must have 'description_file' as key"
        assert isinstance(dict_params['model_str'], str), "'model' must be a string"
        assert isinstance(dict_params['params_name'], list), "'params_name' must be a list"
        assert isinstance(dict_params['bo'], tuple), "'bo' must be a tuple"
        assert len(dict_params['bo']) == 2, "'bo' must be a tuple of 2 values"
        assert isinstance(dict_params['bo'][0], tuple), "'bo' in 1st position must be a tuple"
        assert isinstance(dict_params['bo'][1], tuple), "'bo' in 2nd position must be a tuple"
        aux_cond = len(dict_params['bo'][0]) == len(dict_params['params_name']) and \
                   len(dict_params['bo'][1]) == len(dict_params['params_name'])
        assert aux_cond, "length of the 2 tuples of 'bo' must be equal to length of 'params_name'"
        aux_cond = len(dict_params['ini']) == len(dict_params['params_name'])
        assert aux_cond, "length of 'ini' must be equal to length of 'params_name'"
        aux_cond = True
        aux_cond2 = True
        aux_cond3 = True
        for i in range(len(dict_params['ini'])):
            tpl = dict_params['ini'][i]
            bo_aux = dict_params['bo']
            if not isinstance(tpl, tuple):
                aux_cond = False
            else:
                if len(tpl) != 2:
                    aux_cond2 = False
                else:
                    if tpl[0] < bo_aux[0][i] or tpl[1] > bo_aux[1][i]:
                        print('i', i, 'ini[0]', tpl[0], 'bo[0]', bo_aux[0][i], 'ini[1]', tpl[1], 'bo[1]', bo_aux[1][i])
                        aux_cond3 = False
        assert aux_cond, "elements params_name['ini'] must be tuples"
        assert aux_cond2, "elements params_name['ini'] must have length 2"
        assert aux_cond3, "elements params_name['ini'] must have contained within the boundaries of 'bo'"
        aux_cond = len(dict_params['type_sample_param']) == len(dict_params['params_name'])
        assert aux_cond, "length of params_name['type_sample_param'] must be equal to length of 'params_name'"
        aux_cond = True
        for k in dict_params['type_sample_param']:
            if k not in ['su', 'sn']:
                aux_cond = False
        assert aux_cond, "elements of params_name['type_sample_param'] must be 'su' or 'sn'"
        assert isinstance(dict_params['ODE_mode'], str), "dict_params['ODE_mode'] must be a string"
        assert dict_params['ODE_mode'] in self.types_ODE, "dict_params['ODE_mode'] must be in " + str(self.types_ODE)
        assert isinstance(dict_params['optimizer_mode'], str), "dict_params['optimizer_mode'] must be a string"
        assert dict_params['optimizer_mode'] in self.types_optimizer, ("dict_params['optimizer_mode'] must be in " +
                                                                       str(self.types_optimizer))
        assert isinstance(dict_params['ind_experiment'], int), "dict_params['ind_experiment'] must be an int"
        assert dict_params['ind_experiment'] >= 0, "dict_params['ind_experiment'] must be positive or zero"
        assert isinstance(dict_params['only_spikes'], bool), "'only_spikes' must be a boolean"
        assert os.path.isdir(dict_params['path']), "Folder %s does not exist" % dict_params['path']
        assert isinstance(dict_params['description_file'], str), "dict_params['description_file'] must be a string"

