import yaml

global cfg
if 'cfg' not in globals():
    with open("/Users/tracy/Desktop/Github/nts/src/config.yml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

def process_args(args):
    for k in cfg:
        cfg[k] = args[k]
    return