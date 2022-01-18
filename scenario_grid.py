import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml

# The following inputs need to be in CLI
output_path = Path("mm_use")
suffix = 'exp11d'

scenario_grid = {'arrival_rate': np.arange(.3, 1.3, 0.1), 'mean_los_obs': np.atleast_1d(2.4),
                 'mean_los_ldr': np.atleast_1d(12.0), 'mean_los_csect': np.atleast_1d(1.0),
                 'mean_los_pp_noc': np.atleast_1d(48.0), 'mean_los_pp_c': np.atleast_1d(72.0),
                 'c_sect_prob': np.atleast_1d(0.25),
                 'num_erlang_stages_obs': np.atleast_1d(1), 'num_erlang_stages_ldr': np.atleast_1d(1),
                 'num_erlang_stages_csect': np.atleast_1d(1), 'num_erlang_stages_pp': np.atleast_1d(1),
                 'cap_obs': np.atleast_1d(1000), 'cap_ldr': np.atleast_1d(16), 'cap_pp': np.atleast_1d(75)}

# Create scenario lists from the grid specs above
scenario_grid_lists = {key: value.tolist() for key, value in scenario_grid.items()}

json_file_path = Path(output_path, f'scenario_grid_{suffix}.json')
yaml_file_path = Path(output_path, f'scenario_grid_{suffix}.yaml')

with open(json_file_path, 'w') as f_json:
    json.dump(scenario_grid_lists, f_json, sort_keys=False, indent=4)

with open(yaml_file_path, 'w') as f_yaml:
    yaml.dump(scenario_grid_lists, f_yaml, sort_keys=False)
