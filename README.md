# SynthACticBench

Welcome to SynthACticBench! This repository contains the code for a set of benchmarking functions designed to evaluate compare algorithm configurators by ability. Instead of providing a general-purpose benchmark, SynthACticBench isolates and targets key abilities that configurators need to succeed in algorithm configuration tasks.

Algorithm configurators are powerful tools for automatically optimizing algorithm parameters, but understanding their strengths and weaknesses can be challenging. Real-world benchmarks often combine many factors, making it difficult to pinpoint exactly which abilities—like handling hierarchical configuration spaces or optimizing noisy objective functions—contribute to a configurator's success.

SynthACticBench addresses this by offering:

* Ability-oriented evaluation: Each benchmark isolates a specific property of algorithm configuration problems, such as handling conditional parameters or navigating rugged fitness landscapes. Fig. 1 shows all of the capabilities that can be investigated with SynthACticBench.
* Controlled and reproducible experiments: With transparent and highly configurable benchmarks, you can systematically test configurators under well-defined conditions. Each benchmark is defined by a configuration file, in which can specify any parameter passed to the benchmark's unique function. For all benchmarks, you can freely set the dimensionality and the seed.

![Fig.1 Capabilities evaluated in SynthACticBench](https://github.com/user-attachments/assets/cb4c1651-5392-4b0a-8a51-d76ed65b14b2)
-> TODO: Mit neuer Version updaten!

So this library provides a framework for answering questions like:

* Does this configurator deal well with hierarchical or conditional parameter spaces?
* How well can it adapt to noisy objective functions?
* Can it effectively idenfity a few relevant parameters from a large number of mostly irrelevant parameters?

With SynthACticBench, you can move beyond general performance metrics and dive deeper into the specific capabilities that make a configurator effective. This library was developed in collaboration with [CARPS-S](https://github.com/automl/CARP-S).

## Installation
```bash
# Create conda env
conda create -n synthacticbench python=3.12 -c conda-forge

# Activate env
conda activate synthacticbench

# Clone repo
git clone git@github.com:annaelisalappe/SynthACticBench.git
cd SynthACticBench

# Install 
pip install -e .
```

## Running an optimizer from carps
The command for running an optimizer, e.g., random search from carps on SynthACticBench looks as follows:
```bash
python -m carps.run 'hydra.searchpath=[pkg://synthacticbench/configs]' +optimizer/randomsearch=config +problem/SynthACticBench=C5-ShiftingDomains
```
The breakdown of the command:
- `'hydra.searchpath=[pkg://synthacticbench/configs]'`: Let hydra know where to find the configs of the SynthACticBench package. For this, `synthactic` needs to be installed. In our case, this is: pkg://synthacticbench/configs. 
- `+optimizer/randomsearch=config`: select an optimizer from `carps`. Follows the config folder structure in `carps`. Beware, for other optimizers you need to install dependencies (check the [repo](https://github.com/automl/CARP-S)). To reproduce the experiments we conducted in our initial study, please refer to the [SMAC](https://github.com/automl/SMAC3) installation instructions and the [iracepy-tiny](https://github.com/Saethox/iracepy-tiny/tree/master) installation instructions. 
- `+problem/SynthACticBench=C5-ShiftingDomains`: Select a problem. Follows the configs folder structure in this package, starting from `synthacticbench/configs`. Every benchmark is defined in a configuration file inside this folder. In this file you can specify the parameters for each function, set the seed, as well as the number of trials to run, etc. 

Of course, you can also specify the run dir etc.
For more hydra overrides and parallelization, check their [docs/tutorials](https://hydra.cc/docs/advanced/override_grammar/basic/).
