import os
import re
import json
from termcolor import colored

def read_jsonl(jsonl_path):
    """ returns a list of dictionaries """
    with open(jsonl_path) as f:
        lst = f.readlines()
        lst = [
            json.loads(line.strip()) for line in lst if line.strip()
        ]
    return lst

def read_json(json_path):
    """ returns a list of dictionaries """
    with open(json_path) as f:
        lst = f.read()
    return json.loads(lst)

def chunks(list_of_elements, batch_size):
    """ Yield successive batch-sized chunks from list_of_elements. """
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i:i + batch_size]

def filter_inputs(input_lst, output_path, input_field: str = "question"):
    filtered_input_lst = input_lst[:]

    ## ---- filter out already-generated questions ---
    if os.path.isfile(output_path):
        output_file = read_jsonl(output_path)
        output_lst = [dct[input_field] for dct in output_file]
        for input in input_lst:
            if input[input_field] in output_lst:
                filtered_input_lst.remove(input)
    return filtered_input_lst


def filter_inputs_two_fields(input_lst, output_path, input_fields: list = ["Reddit ID", "appraisal_dimension_id"]):
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


def extract_evaluation_score(text):
    """ Return the evaluation score from the critic model on the intermediate responses. """
    pattern = r'\[RESULT\]\s*([1-5])'
    match = re.search(pattern, text)

    pattern_2 = r'\[SCORE\]\s*([1-5])'
    match_2 = re.search(pattern_2, text)

    pattern_3 = r'\[Score:\s*([1-5])\]'
    match_3 = re.search(pattern_3, text)

    pattern_4 = r'\n\n\[([1-5])\]$'
    match_4 = re.search(pattern_4, text)

    pattern_5 = r'\n\nOverall Score:\s*([1-5])'
    match_5 = re.search(pattern_5, text)

    pattern_6 = r'\[([1-5])\]\Z'
    match_6 = re.search(pattern_6, text)

    if match:
        return int(match.group(1))
    elif match_2:
        return int(match_2.group(1))
    elif match_3:
        return int(match_3.group(1))
    elif match_4:
        return int(match_4.group(1))
    elif match_5:
        return int(match_5.group(1))
    elif match_6:
        return int(match_6.group(1))
    else:
        return None

def extract_evaluation_feedback(text):
    """ Return the evaluation feedback from the critic model on the intermediate responses. """
    return text.split("[RESULT]")[0].strip()

def extract_llama3_evaluation_score(text):
    """ Return the evaluation score from llama3 on the final responses. """
    pattern = r'\[\[(\d{1,2})\]\]'
    match = re.search(pattern, text)

    if match:
        return int(match.group(1))
    else:
        return None


def extract_biggenbench_eval_result(text):
    """ extract the evaluation score for BigGenBench from various LLMs """
    # Pattern to match <score>1-5</score> or <score>[1-5]</score> including decimals, e.g., 2.5
    pattern = r'<score>\[?([1-4](\.\d+)?|5(\.0)?)\]?</score>'

    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    else:
        return None

def extract_biggenbench_eval_feedback(text):
    """ Extract the feedback content for BigGenBench from the given text. """
    # Pattern to match text between <feedback> and </feedback>
    pattern = r'<feedback>(.*?)</feedback>'

    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL to handle multi-line feedback
    if match:
        return match.group(1).strip()  # Strip to remove extra whitespace
    else:
        return None