from utils import *
from scipy.optimize import curve_fit, differential_evolution, NonlinearConstraint

from synaptic_dynamic_models.MSSM import MSSM_model
from synaptic_dynamic_models.LAP import LAP_model
from synaptic_dynamic_models.TM import TM_model


class Fit_params_stp:
    def __init__(self, max_t, sfreq, dict_params):

        self.Input = None

        # model vars
        self.params_name = None
        self.bo = None
        self.ini = None
        self.parameters = None
        self.kwargs_model = None
        self.model_str = None
        self.model_stp = None
        self.fitted_parameters = None
        self.results_optimization = None
        self.types_ODE = ['ODE', 'odeint']
        self.types_optimizer = ['NLS', 'DE']
        self.types_stp = ['MSSM', 'LAP', 'TM']
        self.fit_new_params = None

        # Time vars
        self.sfreq = sfreq
        self.dt = None
        self.time_vector = None
        self.L = None
        self.max_t = None

        #
        self.set_time_vars(max_t)
        self.assert_dict_params(dict_params)
        self.set_model(dict_params)
        # self.sample_ini_params(dict_params)

    def __init__(self, sim_params, dict_params):

        self.Input = None

        # model vars
        self.params_name = None
        self.bo = None
        self.ini = None
        self.parameters = None
        self.kwargs_model = None
        self.model_str = None
        self.model_stp = None
        self.fitted_parameters = None
        self.results_optimization = None
        self.types_ODE = ['ODE', 'odeint']
        self.types_optimizer = ['NLS', 'DE']
        self.types_stp = ['MSSM', 'LAP', 'TM']
        self.fit_new_params = None

        # Time vars
        self.sfreq = None
        self.dt = None
        self.time_vector = None
        self.L = None
        self.max_t = None

        #
        self.set_time_vars(sim_params)
        self.assert_dict_params(dict_params)
        self.set_model(dict_params)
        # self.sample_ini_params(dict_params)

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

    def set_model(self, dict_params):
        """

        Parameters
        ----------
        Returns
        -------
        """
        self.model_str = dict_params['model_str']
        # sim_params = {'sfreq': self.sfreq, 'max_t': self.max_t}
        sim_params = {'sfreq': self.sfreq, 'max_t': self.max_t, 'time_vector': self.time_vector, 'L': self.L}

        # CREATING MODELS
        if self.model_str == "MSSM":
            self.model_stp = MSSM_model()
            self.model_stp.set_simulation_params(sim_params)
        if self.model_str == "LAP":
            self.model_stp = LAP_model()
            self.model_stp.set_simulation_params(sim_params)
        if self.model_str == "TM":
            self.model_stp = TM_model()
            self.model_stp.set_simulation_params(sim_params)

    def sample_ini_params(self, dict_params=None):

        if dict_params is None:
            self.params_name = ['tao_c', 'alpha',  # 'C_0', # 'C0',
                                'V0', 'tao_v',  # 'K_V' 'V_0',
                                'P0',
                                'k_NtV', 'k_Nt', 'tao_Nt',  # 'Nt0', 'Nt_0',
                                'k_EPSP', 'tao_EPSP'  # , 'E_0'
                                ]
            self.bo = ((self.dt + .1 * self.dt, 0.0,  # 0.0, # 0.0,
                        0.0, self.dt + .1 * self.dt,  # 0.0, 0.0,
                        0.0,
                        0.0, 0.0, self.dt + .1 * self.dt,  # 0.0, 0.0,
                        0.0, self.dt + .1 * self.dt  # , 0.0
                        ),
                       (5.0, 1.0,  # 10.0, # 10.0,
                        1.0, 1.0,  # 2000.0, 10.0,
                        1.0,
                        2e4, 1e3, 5.0,  # 10.0, 10.0,
                        2000, 1e-1  # ,100
                        ))
            self.ini = [(2e-1, 3e-2), (2e-1, 3e-2), (0, 10), (0, 10),
                        (0, 1), (2e-1, 3e-2), (500, 200),
                        (0, 10), (0.0, 1.0),
                        (0, 2e4), (0, 1e3), (2e-1, 3e-2), (0, 10), (0, 10),
                        (0, 2000), (2e-2, 3e-4), (0, 100)]
            ini = self.ini
            self.parameters = [sn(ini[0]), sn(ini[1]),  # su(ini[2]), # su(ini[3]),
                               su(ini[4]), sn(ini[5]),  # sn(ini[6]), su(ini[7]),
                               su(ini[8]),
                               su(ini[9]), su(ini[10]), sn(ini[11]),  # su(ini[12]), su(ini[13]),
                               su(ini[14]), sn(ini[15])  # , su(ini[16])
                               ]
        else:
            self.params_name = dict_params['params_name']
            self.bo = dict_params['bo']
            self.ini = dict_params['ini']
            ini = self.ini
            # self.parameters = dict_params['parameters']
            self.parameters = []
            for i in range(len(dict_params['type_sample_param'])):
                k = dict_params['type_sample_param'][i]
                if k == 'su':
                    self.parameters.append(su(ini[i]))
                if k == 'sn':
                    self.parameters.append(sn(ini[i]))
            print("Fit_params_stp, sample_ini_params, ", self.parameters)
        return np.array(self.parameters, dtype='float64')

    def set_input(self, input_vector):
        self.Input = input_vector

    def eval_high_freq_response(self, max_freq, params_name, *parameters):
        """
        Evaluation of MSSM response at high frequencies (max_freq)
        Parameters
        ----------
        max_freq
        params_name
        parameters

        Returns
        -------
        (boolean) whether the parameters lead to the exhaustion of vesicles (the weird behavior) or not
        """
        # Simulation parameters
        sfreq = self.sfreq
        r = max_freq
        end_t, max_t = 20.1, 20.2
        time_vector = np.arange(0, max_t, self.dt)
        L = time_vector.shape[0]
        sim_params = {'sfreq': sfreq, 'max_t': max_t, 'L': L, 'time_vector': time_vector}
        # Input vector
        Input = input_spike_train(sfreq, r, end_t)
        Input = np.concatenate((Input, np.zeros(L - Input.shape[0])))
        Input[0] = 0
        # Setting the auxiliar model
        model_stp_aux = MSSM_model(n_syn=1)
        model_stp_aux.set_simulation_params(sim_params)
        name_params = mssm_name_params
        # Creating keyword arguments
        kwargs_model = {'model': model_stp_aux, 'Input': Input, 'params_name': params_name, 'mode': 'ODE',
                        'block_ves_dep': True}
        # Running model
        model_mssm(time_vector, *parameters, **kwargs_model)
        # Evaluate condition of weird behavior
        if len(model_stp_aux.ind_v_minor_to_0) > 0:
            return True
        else:
            return False

    def run_fit(self, ref_vector_, dict_params):

        # Checking input and reference
        assert self.Input is not None, "Input is not defined"
        # assert len(ref_vector_) == len(self.Input), "Input and reference must have the same shape"

        # Checking content of dict_params
        self.assert_dict_params(dict_params)

        # Getting other parameters
        path_to_vars = dict_params['path']
        ODE_mode = dict_params['ODE_mode']
        optimizer_mode = dict_params['optimizer_mode']
        exp_ind_save = dict_params['exp_ind_save']
        only_spikes = dict_params['only_spikes']
        restriction_vesicle_depletion = dict_params['restriction_vesicle_depletion']
        # fit_new_params = dict_params['fit_new_params']
        suf_file = dict_params['description_file']
        no_ini_conditions = False
        output_factor = 1.
        if 'no_initial_cond_mssm' in dict_params.keys():
            no_ini_conditions = dict_params['no_initial_cond_mssm']
        if 'output_factor' in dict_params.keys():
            output_factor = dict_params['output_factor']

        # Escalation of the ref_vector given the output factor
        ref_vector = ref_vector_ * output_factor

        # Creating name of new file
        sufix = "_" + str(exp_ind_save)
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
            ini_params = self.sample_ini_params(dict_params=dict_params)

            # defining kwargs
            self.kwargs_model = {'model': self.model_stp, 'Input': self.Input, 'params_name': self.params_name,
                                 'mode': ODE_mode, 'only spikes': only_spikes, 'block_ves_dep': False}

            def fun_aux_nls(time_vector, *ini_params):
                if self.model_str == "MSSM":
                    eval_model = model_mssm(time_vector, *ini_params, **self.kwargs_model)
                    print("model evaluation: ", rmse(ref_vector, eval_model))
                    # print("reference vector: ",  ref_vector)
                    return eval_model
                elif self.model_str == "LAP":
                    eval_model = model_lap(time_vector, *ini_params, **self.kwargs_model)
                    print("model evaluation: ", rmse(ref_vector, eval_model))
                    return eval_model
                elif self.model_str == "TM":
                    eval_model = model_tm(time_vector, *ini_params, **self.kwargs_model)
                    print("model evaluation: ", rmse(ref_vector, eval_model))
                    return eval_model
                else:
                    assert False, "Error, model not support"

            def fun_aux_de(x):
                eval_model = None
                if self.model_str == "MSSM":
                    eval_model = model_mssm(self.time_vector, *x, **self.kwargs_model)
                    # Check if the optimization process should consider the restriction of not vesicle depletion
                    # in high frequency responses
                    if restriction_vesicle_depletion:
                        # Restriction that avoids the weird behavior of some fitted parameters for high freq. responses
                        cond_mssm = self.eval_high_freq_response(100, self.params_name, *x)
                        # If the weird behavior is shown for the set of parameters 'x', add heavy penalty to
                        # fitness function
                        if cond_mssm:
                            eval_model = (np.mean(eval_model) + eval_model) * 1.1
                    return rmse(ref_vector, eval_model)
                elif self.model_str == "LAP":
                    eval_model = model_lap(self.time_vector, *x, **self.kwargs_model)
                    # print("model evaluation: ", rmse(ref_vector, eval_model))
                    return rmse(ref_vector, eval_model)
                elif self.model_str == "TM":
                    eval_model = model_tm(self.time_vector, *x, **self.kwargs_model)
                    # print("model evaluation: ", rmse(ref_vector, eval_model))
                    return rmse(ref_vector, eval_model)
                else:
                    assert False, "Error, model not support"

            # def fun_constraint_de(x): return x[7] / x[6]

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
                results_de = differential_evolution(fun_aux_de, bounds=bo_de, disp=True)  # , constraints=nlc)
                parametersoln = results_de.x
                self.results_optimization = results_de
            else:
                assert False, "optimization method not recognized"

            # Varibles to save
            dict_to_save = {"params_name": self.params_name, "bounds": self.bo, "range_ini": self.ini,
                            "ini_params": ini_params, "params": parametersoln, "reference": ref_vector,
                            "scaling_factor_ref": output_factor, "res_optimization": self.results_optimization,
                            'block_ves_dep': restriction_vesicle_depletion}

            # **********************************************************************************************************
            # FOR MSSM, RUN FREQUENCY ANALYSIS AND SAVE THE INDICES AND FREQUENCIES OF SOLUTIONS WITH WEIRD BEHAVIOR
            # fa = RUN FREQUENCY ANALYSIS
            # if self.model_str == "TM":
            #     dict_to_save['syn_with_ves_dep_behavior'] = fa.MSSM_syn_with_weird_freq
            # **********************************************************************************************************

            while check_file(aux_file):
                exp_ind_save += 1
                sufix = "_" + str(exp_ind_save)
                aux_file = (path_to_vars + "param_estimation_" + ODE_mode + "_" + optimizer_mode + "_" +
                            self.model_str + "_" + suf_file + sufix + ".pkl")

            # Save the results as a .pkl file in the folder "variables"
            saved_file = open(aux_file, 'wb')
            pickle.dump(dict_to_save, saved_file)
            saved_file.close()

            print("For experiment ", exp_ind_save, ", parameters are: ", parametersoln)
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

            # ref_vector /= output_factor
            output_factor = dict_to_read['scaling_factor_ref']
            # no_ini_con = False
            # if 'no_ini_cond' in dict_to_read.keys():
            #     no_ini_con = dict_to_read['no_ini_cond']
            # Creating model
            # mssm = MSSM_model(p_lap=aux_p, no_ini_cond=no_ini_con)
            # mssm.set_simulation_params(sim_params)

            restriction_vesicle_depletion = False
            if 'block_ves_dep' in dict_to_read.keys():
                restriction_vesicle_depletion = dict_to_read['block_ves_dep']

            self.kwargs_model = {'model': self.model_stp, 'Input': self.Input,
                                 'params_name': dict_to_read['params_name'], 'mode': ODE_mode,
                                 'only spikes': only_spikes, 'block_ves_dep': restriction_vesicle_depletion}
            # ref_vector *= output_factor

        return self.fitted_parameters

    def evaluate_model(self):
        # Evaluating MSSM
        if self.model_str == "MSSM":
            model_mssm(self.time_vector, *self.fitted_parameters, **self.kwargs_model)
        elif self.model_str == "LAP":
            model_lap(self.time_vector, *self.fitted_parameters, **self.kwargs_model)
        elif self.model_str == "TM":
            model_tm(self.time_vector, *self.fitted_parameters, **self.kwargs_model)

    def assert_dict_params(self, dict_params):
        assert isinstance(dict_params, dict), "dict_params must be a Dictionary"
        assert 'model_str' in dict_params.keys(), "dict_params must have 'model_str' as key"
        assert 'params_name' in dict_params.keys(), "dict_params must have 'params_name' as key"
        assert 'bo' in dict_params.keys(), "dict_params must have 'bo' as key"
        assert 'ini' in dict_params.keys(), "dict_params must have 'ini' as key"
        assert 'type_sample_param' in dict_params.keys(), "dict_params must have 'type_sample_param' as key"
        assert 'ODE_mode' in dict_params.keys(), "dict_params must have 'ODE_mode' as key"
        assert 'optimizer_mode' in dict_params.keys(), "dict_params must have 'optimizer_mode' as key"
        assert 'exp_ind_save' in dict_params.keys(), "dict_params must have 'exp_ind_save' as key"
        assert 'only_spikes' in dict_params.keys(), "dict_params must have 'only_spikes' as key"
        # assert 'fit_new_params' in dict_params.keys(), "dict_params must have 'fit_new_params' as key"
        assert 'restriction_vesicle_depletion' in dict_params.keys(), "dict_params must have 'restriction_vesicle_depletion' as key"
        assert 'path' in dict_params.keys(), "dict_params must have 'path' as key"
        assert 'description_file' in dict_params.keys(), "dict_params must have 'description_file' as key"
        assert isinstance(dict_params['model_str'], str), "'model' must be a string"
        assert dict_params['model_str'] in self.types_stp, "dict_params['model_str'] must be in " + str(self.types_stp)
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
        assert isinstance(dict_params['exp_ind_save'], int), "dict_params['exp_ind_save'] must be an int"
        assert dict_params['exp_ind_save'] >= 0, "dict_params['exp_ind_save'] must be positive or zero"
        assert isinstance(dict_params['only_spikes'], bool), "'only_spikes' must be a boolean"
        # assert isinstance(dict_params['fit_new_params'], bool), "'fit_new_params' must be a boolean"
        assert isinstance(dict_params['restriction_vesicle_depletion'], bool), "'restriction_vesicle_depletion' must be a boolean"
        assert os.path.isdir(dict_params['path']), "Folder %s does not exist" % dict_params['path']
        assert isinstance(dict_params['description_file'], str), "dict_params['description_file'] must be a string"
