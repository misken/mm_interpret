import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import json
import yaml


def scenario_grid_to_csv(path_scenario_grid_yaml, _meta_inputs_path):
    """
    Creates obsimpy metainputs csv file from scenario grid YAML file

    Parameters
    ----------
    path_scenario_grid_yaml
    _meta_inputs_path

    Returns
    -------
    None. The metainputs csv file is written to ``meta_inputs_path``.
    """

    with open(path_scenario_grid_yaml, "r") as stream:
        try:
            _scenario_grid = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    input_scenarios = [scn for scn in itertools.product(*[value for key, value in _scenario_grid.items()])]

    cols = list(_scenario_grid.keys())

    input_scenarios_df = pd.DataFrame(input_scenarios, columns=cols)
    num_scenarios = len(input_scenarios_df.index)
    input_scenarios_df.set_index(np.arange(1, num_scenarios + 1), inplace=True)
    input_scenarios_df.index.name = 'scenario'

    # Create meta inputs scenario file to use for simulation runs
    input_scenarios_df.to_csv(_meta_inputs_path, index=True)
    print(f'Metainputs csv file written to {_meta_inputs_path}')


# The following inputs need to be in CLI

output_path = Path("mm_use") # Destination for YAML scenarios file
exp = 'exp11f' # Used to create subdirs and filenames
siminout_path = Path("data/siminout") # Destination for metainputs csv file based on scenarios

# Need to come up with way to make this dict a file driven thing (e.g. JSON or YAML)
scenario_grid = {'arrival_rate': np.linspace(0.2, 0.4, num=10), 'mean_los_obs': np.atleast_1d(48.0),
                 'mean_los_ldr': np.atleast_1d(48.0), 'mean_los_csect': np.atleast_1d(0.0),
                 'mean_los_pp_noc': np.atleast_1d(72.0), 'mean_los_pp_c': np.atleast_1d(72.0),
                 'c_sect_prob': np.atleast_1d(0.0),
                 'num_erlang_stages_obs': np.atleast_1d(1), 'num_erlang_stages_ldr': np.atleast_1d(2),
                 'num_erlang_stages_csect': np.atleast_1d(1), 'num_erlang_stages_pp': np.atleast_1d(4),
                 'cap_obs': np.atleast_1d(30), 'cap_ldr': np.atleast_1d(25), 'cap_pp': np.atleast_1d(35)}

# Make scenario specific subdirectory if it doesn't already exist for writing the meta
# inputs file to
Path(siminout_path, exp, ).mkdir(exist_ok=True)
meta_inputs_path = Path(siminout_path, exp, f'{exp}_obflow06_metainputs.csv')

# Create scenario lists from the grid specs above
scenario_grid_lists = {key: value.tolist() for key, value in scenario_grid.items()}

json_file_path = Path(output_path, f'scenario_grid_{exp}.json')
yaml_file_path = Path(output_path, f'scenario_grid_{exp}.yaml')

with open(json_file_path, 'w') as f_json:
    json.dump(scenario_grid_lists, f_json, sort_keys=False, indent=4)

with open(yaml_file_path, 'w') as f_yaml:
    yaml.dump(scenario_grid_lists, f_yaml, sort_keys=False)
    print(f'Scenario grid YAML file written to {yaml_file_path}')

scenario_grid_to_csv(yaml_file_path, meta_inputs_path)
