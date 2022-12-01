import yaml
import os


class HyperparameterReader:

    def __init__(self, hyperparam_file: str, verbose: bool=True):

        self.verbose = verbose

        hyperparam_file = os.path.join("hyperparameters", hyperparam_file)

        if os.path.isfile(hyperparam_file) and hyperparam_file.endswith('.yaml'):
            self.hyperparam_file = hyperparam_file
        else:
            raise Exception(f"Hyperparameters file {hyperparam_file} does not exist")
    
    def load_param_dict(self):

        if self.verbose:
            print("[I] Loading hyperparameters")

        file = open(self.hyperparam_file, 'r')
        parameter_dict = yaml.safe_load(file)

        if self.verbose:
            if len(parameter_dict) > 0:
                print("[I] Hyperparameters succesfully loaded")
                for key, val in parameter_dict.items():
                    print(f"\t{key}: {val}")
            else:
                print(f"[I] Cannot load hyperparametes. Check {self.hyperparam_file}")

        file.close()

        return parameter_dict


if __name__ == '__main__':

    hyperparameter_loader = HyperparameterReader("hyperparameters.yaml")
    parameter_dict = hyperparameter_loader.load_param_dict()
