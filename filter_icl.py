import json
from IPython import embed

icl_dir = '/home/ubuntu/sifan/data/icl_pred'

name_ls = ['train','val']

for name in name_ls:
    pos = icl_dir + '/' + name +'.json'
    new_string = []
    new_pos = icl_dir + '/new/' + name +'.json'
    with open(pos, 'r') as f:
        json_file = json.load(f)
        for icl in json_file:
            text = icl['ice_prompt'].split('\n')[0]
            if len(text)>0:
                new_string.append(icl)
    with open(new_pos, 'w') as f:
        json.dump(new_string, f)
        
                