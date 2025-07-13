# import libraries
from utils import *
from utils_plot import colors, plot_hist_pdf

# **********************************************************************************************************************
# GLOBAL PARAMETERS
model_str = "MSSM"  # String defining the model to use (e.g. MSSM, TM, LAP)
ind = 0                 # Auxiliar index to define the Synaptic Dynamics mechanism (0: depression, 1: facilitation)
ind_experiment = 4      # Index of experiment to run/load
num_experiments = 1     # Number of experiments to run
prefix = prefix_v[ind]  # string defining a description of the SD model: "depression" or "facilitation"

# Setting vars for multiple fitting experiments
experiment_loop_lims = [ind_experiment, ind_experiment + num_experiments]
sel_param_name = None
selected_params = None
size_pop = 0
if model_str == 'MSSM':
    sel_param_name = params_name_mssm
    selected_params = selected_params_mssm
    size_pop = 150
if model_str == 'LAP':
    sel_param_name = params_name_lap
    selected_params = selected_params_lap
    size_pop = 108
if model_str == 'TM':
    sel_param_name = params_name_tm
    selected_params = selected_params_tm
    size_pop = 100

#
params_matrix_t = np.empty((size_pop * num_experiments, len(selected_params)))
params_matrix_t[:] = np.nan

# **********************************************************************************************************************
# PATHS TO STORE/LOAD DATA
path_reference_data = "../reference_data/"
path_store_models = "../outputs/fitting_freq_response/"

# ******************************************************************************************************************
# Loading data from the saved models
ind_ = 0
energies_DE_t = []
params_name = None
# Going through all experiments
c = 0  # cont_de
aux_energy_DE = []
for i in range(experiment_loop_lims[0], experiment_loop_lims[1]):
    sufix = "_" + str(i)
    aux_file = path_store_models + "param_estimation_" + model_str + "_" + str(ind) + sufix + ".pkl"
    assert aux_file, "File does not exist"

    if os.path.isfile(aux_file):
        # If the file is already saved, then load it
        read_file = open(aux_file, 'rb')
        dict_to_read = pickle.load(read_file)
        assert isinstance(dict_to_read, dict), "uploaded var is not an instance of dict"

        if params_name is None:
            params_name = dict_to_read['params_name']
        if i == experiment_loop_lims[0]:
            if sel_param_name is not None:
                params_name = sel_param_name

        aux_array = dict_to_read['res_optimization'].population
        params_matrix_t[size_pop * c: size_pop * (c + 1), :] = aux_array
        aux_energy_DE.append(dict_to_read['res_optimization']['population_energies'])

        c += 1
energies_DE_t.append(np.ravel(np.array(aux_energy_DE)))
ind_ += 1

# Excluding solutions with atypical frequency response (in case of MSSM) if the information exists in the saved objects
params_matrix_all = np.copy(params_matrix_t)

# **********************************************************************************************************************
# PLOTS
fig2 = None
max_graph = 5
pos_i = [2, max_graph, 1]
hatchs = ['\\\\\\\\', '////']
if model_str == "MSSM":
    fig2 = plt.figure(figsize=(15, 4))
    pos_i = [2, max_graph, 1]
if model_str == "TM":
    fig2 = plt.figure(figsize=(15, 1.89))
    pos_i = [1, max_graph, 1]
if model_str == "LAP":
    fig2 = plt.figure(figsize=(15, 8))
    pos_i = [3, max_graph, 1]
fig2.suptitle("Distribution of parameters found by DE for the " + model_str, size=24, c='gray')

a = []
for p in range(1, params_matrix_t.shape[1] + 1):

    pos_i[2] = p
    pos = tuple(pos_i)
    a = params_matrix_t[:, p - 1].T

    if p > 1: labels = None
    else: labels = prefix

    plot_hist_pdf(a, labels=labels, title=params_name[p - 1], fig=fig2, pos=pos, returnAx=True, sizeTitle=20,
                  sizeLabels=18, colors=colors, hatchs=hatchs, xAxisSci=True, yAxisFontSize=10)

fig2.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

if model_str == "TM":
    fig2.subplots_adjust(bottom=0.16, right=0.98, top=0.69, left=0.05)
    fig2.legend(loc=4, bbox_to_anchor=(0.98, 0.2), fontsize=14)
if model_str == "MSSM":
    fig2.subplots_adjust(bottom=0.1, right=0.98, top=0.8, left=0.05)
    fig2.legend(loc=8, bbox_to_anchor=(0.91, 0.13), fontsize=14)

# plt.savefig('plots/paper ESANN/poster_params_DE_' + model + '.svg', format='svg')


