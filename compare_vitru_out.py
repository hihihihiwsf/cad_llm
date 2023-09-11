import json

from IPython import embed
import ast

from geometry.visualization import visualize_sample_cv,visualize_batch


def compute_metric(string_completed_entities, string_label_entities):
    
    top1_full, top1_ent = 0, 0
    
    label_entities = sorted(string_label_entities)
    completed_entities = sorted(string_completed_entities)
    
    if len(completed_entities) ==0 or len(label_entities) ==0:
            return 0,0,0,0
    
    if label_entities == completed_entities:
        top1_full=1
    
    if not completed_entities:
        return top1_full, 0, 0, 0
    
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


def circle_to_four_points(circle):
    x,y,r = circle
    # Right point
    x1, y1 = x + r, y
    # Top point
    x2, y2 = x, y + r
    # Left point
    x3, y3 = x - r, y
    # Bottom point
    x4, y4 = x, y - r

    return ((x1, y1), (x2, y2), (x3, y3), (x4, y4))

def find_circle_center(x0, y0, x1, y1, x2, y2):
    D = 2 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))
    try:
        Ux = ((x0**2 + y0**2) * (y1 - y2) + (x1**2 + y1**2) * (y2 - y0) + (x2**2 + y2**2) * (y0 - y1)) / D
        Uy = ((x0**2 + y0**2) * (x2 - x1) + (x1**2 + y1**2) * (x0 - x2) + (x2**2 + y2**2) * (x1 - x0)) / D
    except:
        return x1, y1
    
    return Ux, Uy

def arc_midpoint(arc):
    # Find center of circle
    x0, y0, x1,y1, x2, y2 = arc

    cx, cy = find_circle_center(x0, y0, x1, y1, x2, y2)
    
    # Determine the midpoint of the arc
    mx = (x0 + x2) / 2
    my = (y0 + y2) / 2
    # Direction vector from center to midpoint of endpoints
    dx, dy = mx - cx, my - cy
    # Radius of the circle
    r = ((x0 - cx)**2 + (y0 - cy)**2)**0.5
    # Normalize direction vector
    len_d = (dx**2 + dy**2)**0.5
    try:
        dx /= len_d
        dy /= len_d
    except:
        dx = 0
        dy= 0
    # Compute the new midpoint inside the arc
    mx = cx + dx * r
    my = cy + dy * r
    return ((x0, y0), (mx, my), (x2,y2))
    
def convert_circle(string_ent):
    le = len(string_ent)
    result = any([item < 0 for ent in string_ent for item in ent])
    if result:
        embed()
    new_string = []
    for idx in range(le):
        if len(string_ent[idx])==3:
            new_ent = circle_to_four_points(string_ent[idx])
            new_string.append(new_ent)
        elif len(string_ent[idx]) == 6:
            new_ent = arc_midpoint(string_ent[idx])
            new_string.append(new_ent)
        elif len(string_ent[idx])==2:
            continue
        elif len(string_ent[idx])==4:
            x1,y1,x2,y2 = string_ent[idx]
            new_ent = ((x1,y1),(x2,y2))
            new_string.append(new_ent)
        else:
            new_string.append(string_ent[idx])
    return new_string

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def save_entities(entity, name):
    np_image = render_sketch_opencv(entity, size=512, quantize_bins=64)
    pil_image = np_image[:, :, ::-1]  # BGR to RGB
    img = Image.fromarray(pil_image, mode='RGB')
    # img.save(name)  
    return pil_image

from geometry.parse import get_curves, get_point_entities
def visua_vitru(prefix, output_entities, labels):
    le = len(labels)
    for idx in range(le):
        _prefix = ast.literal_eval(prefix[idx])
        completed = ast.literal_eval(output_entities[idx])

        label = ast.literal_eval(labels[idx])
        completed = completed[len(_prefix):]
        
        top1_full, top1_ent, _, _ = compute_metric(completed, label)
        
        # if top1_full==0 and top1_ent ==1 and len(label)>5:
        _prefix = convert_circle(_prefix)
        completed = convert_circle(completed)
        label = convert_circle(label) 
            
        embed()
        cad_out = '1,13,1,32;1,13,64,13;1,32,63,32;46,51,53,51;63,32,64,13;'
        cadllm = get_point_entities(cad_out)
    
        img1 = save_entities(_prefix, 'outputs/vitru_prefix.png')
        img2 = save_entities(label, "outputs/vitru_label.png")
        img3 = save_entities(completed, "outputs/vitru_completed.png")
        img4 = save_entities(cadllm, "outputs/vitru_completed.png")
        
        fig, axarr = plt.subplots(2, 2)

        axarr[0, 0].imshow(img1, cmap='gray')
        axarr[0, 0].set_title('input')
        axarr[0, 1].imshow(img2, cmap='gray')
        axarr[0, 1].set_title('ground_truth')
        axarr[1, 0].imshow(img3, cmap='gray')
        axarr[1, 0].set_title('output_vitru')
        axarr[1, 1].imshow(img4, cmap='gray')
        axarr[1, 1].set_title('output_cadllm')

        # Removing axis ticks for better visualization
        for ax in axarr.ravel():
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'outputs/vitru_test_{idx}.png')
        embed()
        
from geometry.visualization import render_sketch_opencv              


if __name__ == '__main__':
    
    out_ent_path = '/home/ubuntu/vitruvion/new_test_completed.json'
    out_label_path = '/home/ubuntu/vitruvion/new_test_label.json'
    out_prefix_path= '/home/ubuntu/vitruvion/new_test_input.json'


    with open(out_ent_path,'r') as f:
        output_entities = json.load(f)

    with open(out_label_path, 'r') as f:
        labels = json.load(f)

    with open(out_prefix_path, 'r') as f:
        prefix = json.load(f)
    
    visua_vitru(prefix, output_entities, labels)