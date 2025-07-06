import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--t', type=int)
cmd_args = parser.parse_args()

with open(cmd_args.data_path, 'r') as file:
    doc = yaml.load(file, Loader=yaml.FullLoader)
    doc['AE']['t'] = cmd_args.t
with open(cmd_args.data_path,'w+') as file:
    yaml.dump(doc, file)