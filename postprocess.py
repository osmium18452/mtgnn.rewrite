import json
import os
import pickle

for root, data_list, file_list in os.walk('save/finance.sup/worst'):
    for file in file_list:
        if file.endswith('.json'):
            result = json.load(open(os.path.join(root, file), encoding='utf8'))
            print(result['dataset'], result['output_len'], result['mse'], result['mae'], result['save_step'], sep=',')
