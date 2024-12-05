## Description
In this repository, a pipeline is presented to search for 
parameters of Synaptic Dynamics (SD) models, with the aim of simulating 
experimental _in vitro_ recordings of real synapses. The pipeline have 
different components: an input signal, a reference signal, a SD model 
and an implementation of Differential Evolution (DE).
A diagram of the pipeline is depicted here:

![This is the pipeline](pipeline_.png)

### Input and reference signals
The input and reference signals are the information needed from the _in vitro_
experiments. They must have the same shape in order to run the pipeline.

### Synaptic Dynamics model
This repository implements three SD models: the Tsodyks-Markram (TM) model [[1]](http://www.scholarpedia.org/article/Short-term_synaptic_plasticity),
the Modified Stochastic Synaptic Model (MSSM) [[2]](https://ul.qucosa.de/landing-page/?tx_dlf[id]=https%3A%2F%2Ful.qucosa.de%2Fapi%2Fqucosa%253A11334%2Fmets),
and the Lee-Anton-Poon model (LAP) [[3]](https://link.springer.com/10.1007/s10827-008-0122-6).
All three models inherits from a superclass called SynDynModel, which can be 
considered as the template to implement other SD models.

### Differential Evolution 
De is a population optimization technique, part of the family of Evolutionary
Algorithms. The [_scikit-learn_](https://scikit-learn.org/stable/) implementation
of DE is used in this pipeline. For more information about that implementation,
please follow this [link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).

## How to use the pipeline
The class Fit_params_SD implements the pipeline, it requires five arguments:
1. input_signal: A 1D numpy array with shape _N_, corresponding to the input 
signal of the _in vitro_ experiment.
2. reference_signal: a 1D numpy array with shape _N_, corresponding to the
postsynaptic response of the _in vitro_ experiment, given the input_signal.
3. model_SD: An object of type SynDynModel, specifying one model of Synaptic
Dynamics. In this repository, three models are implemented for general purposes:
MSSM, TM and LAP models. Any other model can be implemented by inheriting from 
the superclass SynDynModel.
4. dict_params: a dictionary containing the following keys:  
   'model_str': A string defining a convention for the SD model. It is suggested
   to use an abbreviation of the model (e.g. 'MSSM', 'TM', or 'LAP').  
   'params_name': params_name_mssm,  
   'bo':   
   'ODE_mode': 'ODE',  
   'ind_experiment': 0,  
   'only_spikes': False,  
   'path': self.path_outputs,  
   'description_file': description,  
    'output_factor'
5. DE_params: a python list, whose order must follow the arguments of the 
   scikit-learn implementation of Differential Evolution. Here a summary of 
   these arguments:  
   strategy='best1bin'  
   generations=1000  
   popsize=15  
   tol=0.01  
   mutation=(0.5, 1)  
   recombination=0.7  
   seed=None  
   callback=None
   disp=True  
   polish=True  
   init='latinhypercube'  
   atol=0  
   updating='immediate'  
   workers=1  
   constraints=()  
   x0=None  
   integrality=None
   vectorized=False


