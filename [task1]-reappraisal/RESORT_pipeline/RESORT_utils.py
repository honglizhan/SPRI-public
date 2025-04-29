import os
import json
import pandas as pd
from termcolor import colored


def read_json(json_path):
    """ returns a list of dictionaries """
    with open(json_path) as f:
        lst = f.read()
    return json.loads(lst)

def get_prompts(path_to_appraisal_questions, path_to_reappraisal_guidance):
    dimensions_df = pd.DataFrame(index = range(1, 25))
    dim_files = [path_to_appraisal_questions, path_to_reappraisal_guidance]

    for dim_file_name in dim_files:
        with open(dim_file_name) as dim_file:
            raw_txt = dim_file.read()
            if 'dim_name' not in dimensions_df.columns:
                dimensions_df['dim_name'] = pd.Series(raw_txt.split('\n'), index = range(1, 25)).apply(lambda line: line[:line.find('=') - 1])
            dimensions_df[dim_file_name.split("/")[-1].split(".")[0]] = pd.Series(raw_txt.split('\n'), index = range(1, 25)).apply(lambda line: line[line.find('=') + 1:])
    #display(dimensions_df)
    return dimensions_df

def filter_inputs(input_lst, output_path, input_fields: list = ["Reddit ID", "appraisal_dimension_id"]):
    assert len(input_fields) == 2, "input_fields must be a list of two fields."
    filtered_input_lst = input_lst[:]

    ## ---- filter out already-generated questions ---
    if os.path.isfile(output_path):
        print (colored(f"Output file found at {output_path}. Filtering out already generated questions.", "red"))
        output_file = read_jsonl(output_path)
        output_tuples = [(dct[input_fields[0]], dct[input_fields[1]]) for dct in output_file]
        for input in input_lst:
            input_tuple = (input[input_fields[0]], input[input_fields[1]])
            if input_tuple in output_tuples:
                filtered_input_lst.remove(input)
        print (colored(f"Filtered out {len(input_lst) - len(filtered_input_lst)} questions.", "green"))
    else:
        print (colored(f"Output file not found at {output_path}.", "red"))
    return filtered_input_lst


def read_jsonl(jsonl_path):
    """ returns a list of dictionaries """
    with open(jsonl_path) as f:
        lst = f.readlines()
        lst = [
            json.loads(line.strip()) for line in lst if line.strip()
        ]
    return lst