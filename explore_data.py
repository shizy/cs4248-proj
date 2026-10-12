import json

def get_dict_structure(d):
    if isinstance(d, dict):
        return {k: get_dict_structure(v) for k, v in d.items()}
    elif isinstance(d, list):
        if d:
            return [get_dict_structure(d[0])]
        return []
    else:
        return type(d).__name__

if __name__ == "__main__":
    with open("./data/train-v1.1.json", "r") as file:
        train_data_dict = json.load(file)

    print(get_dict_structure(train_data_dict))