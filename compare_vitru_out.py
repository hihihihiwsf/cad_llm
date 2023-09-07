import json

from IPython import embed
import ast

from geometry.visualization import visualize_sample_cv,visualize_batch


out_ent_path = '/home/ubuntu/vitruvion/res_entites.json'
out_label_path = '/home/ubuntu/vitruvion/res_label.json'


with open(out_ent_path,'r') as f:
    output_entities = json.load(f)

with open(out_label_path, 'r') as f:
    labels = json.load(f)



def compute_metric(string_completed_entities, string_label_entities):
    
    top1_full, top1_ent = 0, 0
    
    label_entities = sorted(string_label_entities)
    completed_entities = sorted(string_completed_entities)
    
    if len(completed_entities) ==0 or len(label_entities) ==0:
            return 0,0,0
    
    if label_entities == completed_entities:
        top1_full=1
    
    if not completed_entities:
        return top1_full, 0
    
    eps =1e-6
    TP=0
    validity = 0
    for first_entity in completed_entities:
        if len(first_entity) ==4 or len(first_entity)==6 or len(first_entity)==2 or len(first_entity)==3:
            validity +=1 
        if first_entity and first_entity in label_entities:
            top1_ent = 1
            TP+=1
    FP = len(completed_entities) - TP
    FN = len(label_entities) - TP
    precision = (TP / (TP + FP)) + eps
    recall = (TP / (TP + FN)) + eps
    
    f1= 2 * precision * recall / (precision + recall)
    validity = validity / len(completed_entities)
    return top1_full, top1_ent, validity, f1

def visua_vitru(output_entities, labels):
    le = len(labels)
    for idx in range(le):
        completed = ast.literal_eval(output_entities[idx])
        label = ast.literal_eval(labels[idx])
        top1_full, top1_ent, _, _ = compute_metric(completed, labels)
        
        if top1_full==0 and top1_ent ==1:
            if len(labels[idx]>5):
                fig = visualize_batch(input_curves=completed, label_curves=labels,
                              sample_curves=completed, box_lim=67)
                fig_path =  f"outputs/vitru_test_{idx}.png"
                fig.savefig(fig_path)
                


        
        