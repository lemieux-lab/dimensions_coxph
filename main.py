# base
import pdb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )

import experiments.parsers as parsers
args = parsers.parse_arguments()
#FE_args = parsers.parse_FE_arguments()
class ExperimentLoader:
    def __init__(self) -> None:
        # from experiments.exp_1 import v_6 as exp_1_6
        # from experiments.exp_2 import v_1 as exp_2_1
        # from experiments.exp_2 import v_3 as exp_2_3
        # from experiments.plots_1 import v_0 as plot_1_0
        # from experiments.exp_3 import v_0 as exp_3_0
        # from experiments.exp_4 import v_2 as exp_4_2
        # from experiments.exp_4 import v_1 as exp_4_1
        from experiments import experiment1
        from experiments import experiment2
        from experiments import poster 

        self.experiment_dict = {
            "1": experiment1,
            "2": experiment2,
            "poster": poster,
        }
    def load_experiment(self, exp_vn):
        """Load and returns an Experiment Object"""
        return self.experiment_dict[exp_vn]

def main():
    # custom
    e = ExperimentLoader().load_experiment(args.EXP)
    e.run(args)
    # import experiments.experiments_FE as FE_exp
    # FE_exp.run_experiment(FE_args)
    #import experiments.experiments_PCA as PCAExperiments
    #PCAExperiments.experiment_1()
if __name__ == "__main__":
    main()
