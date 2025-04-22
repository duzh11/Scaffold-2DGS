import os
import glob
import json
import wandb
import pandas as pd
import argparse
from argparse import Namespace

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', \
              'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

def summary_results(result_dirs, scene_lis, json_name = 'vis/results.json', extension_name = 'summary', wandb=None):
    if len(scene_lis) == 0:
        print("No scenes to summarize.")
        return None
    else:
        print(f"Summarize {json_name}: " + ", ".join(scene_lis))

    # Initialize lists to store the data
    data = []

    # Read each JSON file and extract the metrics
    for scene in scene_lis:
        # find the result file
        result_file = os.path.join(result_dirs, scene, json_name)
    
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # Extract the metrics
        result_lis = {'Scene' : scene}
        result_lis.update(result)
        
        # Append the data to the list
        data.append(result_lis)

        # write into wandb
        if wandb is not None:
            for key, value in result.items():
                wandb.log({f'{scene}/{key}': value})

    # Calculate average metrics
    df = pd.DataFrame(data)

    average_lis = {'Scene' : 'average'}
    for key in result:
        average_lis[key] = df[key].mean()
        
        if wandb is not None:
            wandb.log({f'{extension_name}/{key}': df[key].mean()}, commit=False)
    
    data.append(average_lis)

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Sort the DataFrame by experiment name
    df = df.sort_values('Scene')

    # Save the table to a CSV file
    csv_path = os.path.join(result_dirs, json_name.split('/')[0])
    os.makedirs(csv_path, exist_ok=True)
    csv_file = (json_name.split('/')[1] ).split('.')[0] + f'_{extension_name}.csv'
    df.to_csv(f'{csv_path}/{csv_file}', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results from JSON files.")
    parser.add_argument("--model_path", "-m", help="model path", default="./exps/full/DTU")
    parser.add_argument('--use_wandb', action='store_true', default=False)
    args = parser.parse_args()

    with open(os.path.join(args.model_path, dtu_scenes[0], 'cfg_args')) as cfg_file:
        cfgfile_string = cfg_file.read()
    args_cfgfile = eval(cfgfile_string)

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="DTU",
            name=f'scaffold-2dgs-{args.model_path.split("/")[-2]}-{args.model_path.split("/")[-1]}',
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args_cfgfile)
        )
    else:
        wandb = None

    summary_results(args.model_path, dtu_scenes, json_name = 'vis/results.json', wandb=wandb)