import numpy as np  
import json

train_file = '/Tmp/sifan/cad/data/sg_string_v5_ascii_max64/train.json'
test_file = '/Tmp/sifan/cad/data/sg_string_v5_ascii_max64/test.json'
val_file = '/Tmp/sifan/cad/data/sg_string_v5_ascii_max64/val.json'

with open(train_file, 'r') as f:
    train_data = json.load(f)

with open(test_file, 'r') as f:
    test_data = json.load(f)

with open(val_file, 'r') as f:
    val_data = json.load(f)   
    
    
model_batch={}
import json
import ast
import numpy as np
from compare_vitru_out import convert_circle, save_entities, center_and_scale
from preprocess.preprocessing import center_vertices
from geometry.visualization import visualize_batch, visualize_sample, visualize_sample_cv, visualize_sample_handraw2
with open('/u/wusifan/vitruvion_autocomplete_new_test_prefix.json', 'r') as f:
    test_prefix = json.load(f)
with open('/u/wusifan/vitruvion_autocomplete_new_test_output.json', 'r') as f:
    test_label = json.load(f)

idx = np.arange(0, 50)
point_input_strings = [ast.literal_eval(pre) for pre in test_prefix]
point_label = [ast.literal_eval(pre) for pre in test_label]
    

point_inputs = [convert_circle(sketch) for sketch in point_input_strings]
point_labels =[convert_circle(sketch) for sketch in point_label]

from IPython import embed

# prefix_img = visualize_sample_cv(point_inputs[100:150], box_lim=64+3)
# label_img = visualize_sample_cv(point_labels[100:150], box_lim=64+3)
# for i in range(30):
#     label_img[i].save(f'outputs/{i}_label.png')
#     prefix_img[i].save(f'outputs/{i}_prefix.png')

saved_idx = [3,6,13,17,28,29, 51, 58, 63, 66, 67, 69,80,82, 86, 91, 92,93,101,110,113,126,127,129,130,132]
saved_inputs=[]
saved_label=[]
input_strings = []
label_strings = []
for i in saved_idx:
    saved_inputs.append(point_inputs[i])
    saved_label.append(point_labels[i])
    input_strings.append(';'.join(','.join(str(item) for pair in tup_set for item in pair) for tup_set in point_inputs[i]) + ';')
    label_strings.append(';'.join(','.join(str(item) for pair in tup_set for item in pair) for tup_set in point_inputs[i]) + ';')
prefix_img = visualize_sample_handraw2(saved_inputs, box_lim=64+3)
label_img = visualize_sample_cv(saved_label, box_lim=64+3)
embed()

for i in range(len(saved_idx)):
    label_img[i].save(f'pdfs/{i}_label.pdf', 'PDF', resolution=100.0)
    prefix_img[i].save(f'pdfs/{i}_prefix.pdf', 'PDF', resolution=100.0)
with open('pdfs/input_strings.json', 'w') as f:
    json.dump(input_strings, f)
with open('pdfs/label_strings.json', 'w') as f:
    json.dump(label_strings, f)
    
tokenized_input = tokenizer(input_strings, max_length=args.max_length, padding=True, truncation=True, return_tensors="pt")
model_batch['input_ids'] = tokenized_input.input_ids
model_batch['attention_mask'] = tokenized_input.attention_mask


# ###tokenized_length
# import matplotlib.pyplot as plt
# all_input_lengths = []
# all_output_lengths = []
# for _, input_lengths, output_lengths in sketchdata.val_dataloader():
#         all_input_lengths.extend(input_lengths)
#         all_output_lengths.extend(output_lengths)
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.hist(all_input_lengths, bins=30, color='blue', alpha=0.7)
# plt.title('Input Token Length Distribution')
# plt.xlabel('Length')
# plt.ylabel('Number of Samples')

# plt.subplot(1, 2, 2)
# plt.hist(all_output_lengths, bins=30, color='red', alpha=0.7)
# plt.title('Output Token Length Distribution')
# plt.xlabel('Length')
# plt.ylabel('Number of Samples')

# plt.tight_layout()
# plt.savefig("val_input_output_length.png")

