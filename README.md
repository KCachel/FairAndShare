# Fair-Multi-Criteria-Candidate-Set-Selection
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Corresponding code, and experiments for paper "Fair & Share: Fast and Fair Candidate Set Selections from Multi-Criteria". 


#### Fair&Share and F&S-Fagin's code:

Please see src folder for `FairShare.py` python script for Fair&Share. 

F&S-Fagin's code is in the `fair_fagins.py` script in the baselines folder. 

#### Comparative baselines:

All baselines used are in the baselines folder. 
 - **Fair**. Please see `baseline_zehlikeetal.py`.
 - **DivTopK**. Please see `baseline_stoyanovichetal.py`.
 - **GbG**. Please see `baseline_GBG_Threshold.py`.
 - **Greedy**. Please see `baseline_greedyfmc.py`.



#### Experiments:


##### Proportional Representation Task
For experiments with proportional fairness objectives please see the exp_with_proportional folder. There `study_delta_proportional.py` performs the experiments. All results are then in the exp_with_proportionalfolder. 

##### Equal Representation Task
For experiments with equal fairness objectives please see the exp_with_equal folder. There `study_delta_equal.py` performs the experiments. All results are then in the exp_with_equal folder.

##### Rooney Rule Task
For experiments with the rooney rule please see the exp_with_rooney folder. There `run_rooney.py` performs the experiments comparing all methods. Results are then in the `rooney_taskadult.csv` file.



#### Helper Functions:
Functions to calculate fairness metrics are in the metrics directory. 