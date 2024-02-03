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

def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = min(min(pair) for pair in tuples)
    vert_max = max(max(pair) for pair in tuples)
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center

def circle_to_four_points(circle):
    x,y,r = circle
    x=x+1
    y=y+1
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
    x0=x0+1
    y0=y0+1
    x1=x1+1
    y1=y1+1
    x2=x2+1
    y2=y2+1
    
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
    x0=int(round(x0))
    x2=int(round(x2))
    mx=int(round(mx))
    y0=int(round(y0))
    y2=int(round(y2))
    my=int(round(my))

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
            new_ent = ((x1+1,y1+1),(x2+1,y2+1))
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
        cad_out = '1,11,1,24;1,11,64,11;1,24,64,24;16,24,21,29;21,29,21,53;21,29,43,29;21,53,43,53;32,43,43,29;43,29,43,53;43,29,48,24;48,24,52,38;' #'1,8,1,56;1,8,63,8;1,56,64,56;63,8,63,55;63,8,64,56;' # '1,3,1,61;1,3,63,3;1,61,63,61;63,3,63,60;63,3,63,61;' # '1,13,1,32;1,13,64,13;1,32,63,32;46,51,53,51;63,32,64,13;'
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
    
    import json
    import ast
    import numpy as np
    from IPython import embed
    import matplotlib.pyplot as plt
    from compare_vitru_out import convert_circle, save_entities
    from preprocess.preprocessing import center_vertices, center_and_scale, normalize_and_quantize_vertices
    from geometry.visualization import visualize_batch, visualize_sample, visualize_sample_cv, visualize_sample_handraw2
    with open('/u/wusifan/cadllm/vitruvion/virtruvion_test_prefix.json', 'r') as f:
        test_prefix = json.load(f)
    with open('/u/wusifan/cadllm/vitruvion/virtruvion_test_label.json', 'r') as f:
        test_label = json.load(f)
    with open('/u/wusifan/cadllm/vitruvion/virtruvion_test_output.json', 'r') as f:
        test_output = json.load(f)
    
    def center_and_normalize(part_data, tuple_data):
        if not tuple_data:
            return tuple_data, tuple_data
        vertices = np.array([item for sublist in tuple_data for item in sublist])

        normalized_quantized_vertices = normalize_and_quantize_vertices(vertices)
        normalized_quantized_part_data = []
        output_tuples = []
        start = 0
        for t in tuple_data:
            end = start + len(t)
            output_tuples.append(tuple(map(tuple, normalized_quantized_vertices[start:end])))
            if t in part_data:
                normalized_quantized_part_data.append(tuple(map(tuple, normalized_quantized_vertices[start:end])))
            start = end
        return normalized_quantized_part_data,output_tuples


    point_input_strings = [ast.literal_eval(pre) for pre in test_prefix]
    point_label = [ast.literal_eval(pre) for pre in test_label]
    point_output = [ast.literal_eval(pre) for pre in test_output]
        
    print("start convert circle")
    point_inputs = [convert_circle(sketch) for sketch in point_input_strings]
    point_labels =[convert_circle(sketch) for sketch in point_label]
    point_outputs =[convert_circle(sketch) for sketch in point_output]
    
    point_inputs = point_inputs[1200:]
    point_labels = point_labels[1200:]
    point_outputs = point_outputs[1200:]
    
    print("start center and normalize")
    n_point_inputs = [center_and_normalize(tuple_input, tuple_label)[0] for tuple_input, tuple_label in zip(point_inputs, point_labels)]
    n_point_labels = [center_and_normalize(tuple_input, tuple_label)[1] for tuple_input, tuple_label in zip(point_inputs, point_labels)]
    n_point_outputs = [center_and_normalize(tuple_input, tuple_label)[1] for tuple_input, tuple_label in zip(point_inputs, point_outputs)]
    
    start = 200

    
    # hand_draw = visualize_sample_handraw2(n_point_labels[150:200], box_lim=64+3)
    # shift_fraction = 6 / 128
    # scale = 0.2
    # shear = 8
    # rotation = 8
    # import torchvision
    # img_affine = torchvision.transforms.RandomAffine(
    #     rotation,
    #     translate=(shift_fraction, shift_fraction),
    #     scale=(1 - scale, 1 + scale),
    #     shear=shear,
    #     fill=255)
    # noisy_hand_imgs = [img_affine(imgs) for imgs in hand_draw]
    
    # start = 10000
    # prefix_img = visualize_sample_cv(n_point_inputs[start:start+100], box_lim=64+3)
    # label_img = visualize_sample_cv(n_point_labels[start:start+100], box_lim=64+3)
    # for i in range(100):
    #     label_img[i].save(f'label_outputs/{i}_label.png')
    #     prefix_img[i].save(f'prefix_outputs/{i}_prefix.png')
       #noisy_hand_imgs[i].save(f'noisy_outputs/{i}_noisy.png')


    # saved_idx = [3,6,13,17,28,29, 51, 58, 63, 66, 67, 69,80,82, 86, 91, 92,93,101,110,113,126,127,129,130,132]
    # 2th_saved_idx = [18, 23, 26,32, 37, 49, 52, 60, 75, 97, 118, 124, 126, 136, 138, 141,146,151,157,158,164,160,163,165,173,181,213,216,217,220,224]
    th2_saved_idx = [18, 23, 26,32, 37, 49, 52, 60, 75, 97, 118, 124, 126, 136, 138, 141,146,151,157,158,164,160,163,165,173,181,213,216,217,220,224,236,244,247,252,256,257,264,271,275,276,277,279,280,301,302,303,306,309,315,320,322,323,332,336,340,344,347,356,374,376,377,378,379,398,401,404,407,410,416,417,418,419
    ,425,426,431,436,440,443,455,459,463,464,465,466,471,488,493,492,495,496,503,509,510,514,522,533,535,534
    ,538,540,545,548,561,565,575,584,593,596,601,602,604,605,608,611,615,618,621,624,646,650,651,652,663,667
    ,677,681,682,683,697,698]
    b = [2,5,6,10,13,16,20,33,39,40,42,43,44,51,53,57,61,64,65,67,72,73,88,89,96,97,98,100,102,103,105,113,117,118,
                121,129,133,135,139,150,156,157,163,169,186,193,196,404,407,414,417,429,432,433,434,441,455,469,474,486,491,
                499,519,529,535,557]
    a = [198,195,188,166,164,162,158,155,149,138,135,132,127,123,118,113,110,102,96,93,91,80,78,68,48,45,44,41,40,38,32,26,22,20,15,14,12,5]
    a = [item+10200 for item in a]
    b = [item+10000 for item in b]
    supp_idx = a+b
    embed()
    
    saved_inputs=[]
    saved_label=[]
    saved_output = []
    input_strings = []
    label_strings = []
    print("start append strings")
    for i in supp_idx:
        saved_inputs.append(n_point_inputs[i])
        saved_label.append(n_point_labels[i])
        saved_output.append(n_point_outputs[i])
        input_strings.append(';'.join(','.join(str(item) for pair in tup_set for item in pair) for tup_set in n_point_inputs[i]) + ';')
        label_strings.append(';'.join(','.join(str(item) for pair in tup_set for item in pair) for tup_set in n_point_labels[i]) + ';')
    
    # saved_inputs[58].append(((18, 27), (31, 17), (44, 27)))
    # prefix_img = visualize_sample_cv(saved_inputs[53:59], box_lim=64+3)
    # label_img = visualize_sample_cv(saved_label[53:59], box_lim=64+3)
    # plt.imshow(prefix_img[0])
    # plt.axis('off') 
    # plt.savefig('_img2.pdf')
    # plt.imshow(label_img[0])
    # plt.axis('off') 
    # plt.savefig('_img3.pdf')
    embed()
    print("visulizing samples")
    prefix_img = visualize_sample_cv(saved_inputs, box_lim=64+3)
    label_img = visualize_sample_cv(saved_label, box_lim=64+3)
    #output_img = visualize_sample_cv(saved_output, box_lim=64+3)
    print("saving samples")
    for i in range(len(supp_idx)):
        print(i)
        plt.imshow(label_img[i])
        plt.axis('off') 
        plt.savefig(f'pdfs/{i}_label.pdf', bbox_inches='tight')
        plt.imshow(prefix_img[i])
        plt.axis('off') 
        plt.savefig(f'pdfs/{i}_prefix.pdf', bbox_inches='tight')
        # plt.imshow(output_img[i])
        # plt.axis('off') 
        # plt.savefig(f'pdfs/{i}_vitru.pdf', bbox_inches='tight')
        
    
    with open('pdfs/input_strings2.json', 'w') as f:
        json.dump(input_strings, f)
    with open('pdfs/label_strings2.json', 'w') as f:
        json.dump(label_strings, f)