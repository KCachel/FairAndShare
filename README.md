# Fair-Multi-Criteria-Candidate-Set-Selection
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Corresponding code, and experiments for paper "Fast and Fair Candidate Set Selections from Multi-Criteria". 


#### FMCS and Fair Fagin's code:

Please see src folder for python scripts for FMCS. note, there is a corresponding `FMCS_perfcounts.py` script that contains the algorithm so that it returns various access count metrics).

Fair Fagin's code is in the `fair_fagins.py` script in the baselines folder. 

#### Comparative baselines:

All baselines used are in the baselines folder. 
 - **Fair**. Please see `baseline_zehlikeetal.py`.
 - **DivTopK**. Please see `baseline_stoyanovichetal.py`.
 - **Rank-exposure**. Please see `baseline_guptaetal.py`.
 - **Epsilon-greedy**. Please see `baseline_fengetal.py`.



#### Experiments:


##### Proportional Representation Task
For experiments with proportional fairness objectives please see the exp_with_proportional folder. There `run_proportional.py` performs the experiments comparing all methods, and `study_delta_proportional.py` study FMCS behavior under varying threshold styles and delta parameters. All results are then in the exp_with_proportional\results folder. 

##### Equal Representation Task
For experiments with equal fairness objectives please see the exp_with_equal folder. There `run_equal.py` performs the experiments comparing all methods, and `study_delta_equal.py` study FMCS behavior under varying threshold styles and delta parameters. All results are then in the exp_with_equal\results folder.

##### Rooney Rule Task
For experiments with the rooney rule please see the exp_with_rooney folder. There `run_rooney.py` performs the experiments comparing all methods. Results are then in the `rooney_taskadult.csv` file.
port pdf. The code to reproduce the experiment is in the CSRankings_Experiment: Experiment folder for dataset, and code to run experiment `run_mv_csrankings.py` 

#### Plotting
To keep plotting self-contained the result csvs have been copied over to the plotting directory. `jaccard_heatmaps_for_tasks.R` produces the heatmaps of the jaccard similiarity for the results from each method; it produces the `proportional_maps.pdf` and `equal_maps.pdf` figures. Then `delta_parameter.R` produces plots for each dataset with FMCS under different threshold styles and delta parameters. It's figures are written to the plotting\delta folder.  


#### Helper Functions:
Functions to calculate fairness metrics are in the metrics directory. 