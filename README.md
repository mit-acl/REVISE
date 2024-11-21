# REVISE: Robust Probabilistic Motion Planning in a Gaussian Random Field

## Citation
```bibtex
@misc{rose2024reviserobustprobabilisticmotion,
      title={REVISE: Robust Probabilistic Motion Planning in a Gaussian Random Field}, 
      author={Alex Rose and Naman Aggarwal and Christopher Jewison and Jonathan P. How},
      year={2024},
      eprint={2411.13369},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.13369}, 
}
```

## General Setup
REVISE has been tested with Python 3.12.7 on MacOS and on Ubuntu 20.04.

Installing dependencies (using a virtual environment is recommended):
* `pip install -r requirements.txt`

Replicating paper results:
* `python run_quadrotor_experiment.py` to regenerate belief roadmaps for the single-query and multi-query experiments
* `python run_monte_carlo_simulation.py` to generate random goals for the multi-query experiment and simulate closed-loop trajectories for both experiments
* `python run_metrics_evaluation.py` to evaluate final state MSE, Wasserstein distance between the planned and actual final state distributions, and plan cost for both experiments
* `python plot_results.py` to regenerate the plots used in the paper

## Documentation and Website
Documentation is auto-generated based on the source code and hosted by <a href="https://revise.readthedocs.io/en/latest/">Read the Docs</a>. Our project page is online at <a href="https://acl.mit.edu/REVISE/">https://acl.mit.edu/REVISE/</a>.

## Acknowledgements
This research was supported by the National Science Foundation Graduate Research Fellowship under grant no. 2141064 and by the Draper Scholars program.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
