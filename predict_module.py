from detectron2.utils.logger import setup_logger
setup_logger()

import matplotlib.path as mplPath
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import os, json, cv2, random
import math, time

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

def tidy_data(outputs_1):
    new_classes =[]
    new_bbox = []
    global classes
    classes = np.array(outputs_1["instances"].pred_classes.numpy())
    for i in range(len(classes)):
        pred_class = classes[i]
        new_classes.append(pred_class)
        mask_image = outputs_1["instances"].pred_masks.numpy()
        bbox = bounding(mask_image[i,:,:])
        new_bbox.append(bbox[0])
    new_predict = {'classes':np.array(new_classes),'bbox':new_bbox}
    return new_predict


def bounding(mask):
    total_box = []
    mask = np.clip(mask, 0, 255)# 归一化也行
    mask = np.array(mask,np.uint8)
    (cnts, _) = cv2.findContours(mask.copy(),
                                 cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        
        rect = cv2.minAreaRect(c)
        # calculate coordinate of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box =np.int0(box)
        total_box.append(box)
    return total_box

def is_not_in_other_shrimp2(point,total_matrix):
    x = 0
    for matrix in total_matrix:
        [p1,p2,p3,p4] = matrix
        poly_path = mplPath.Path([p1,p2,p3,p4])
        if poly_path.contains_point(point):
            x+=1
    if x == 1:
        return True
    else:
        return False


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            label_class = anno["name"]
            class_list = ["shrimp","head","tail"]

            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": class_list.index(label_class),
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
balloon_metadata = MetadataCatalog.get("new_shrimp_train")

def rebuild_bbox(img,predict):
    all_shrimp = []
    for i in list(np.where(predict['classes']==0)[0]):
        shrimp_bbox = predict['bbox'][i]
        all_shrimp.append(shrimp_bbox.tolist())

    color = [(255,255,0),(0,255,255),(255,0,255)]

    total_shrimp = {}
    clone = img.copy()

    judge_head = all_shrimp.copy()
    judge_tail = all_shrimp.copy()

    for i in list(np.where(predict['classes']==0)[0]):
        head_bbox = []
        tail_bbox = [] 

        pred_class = classes[i]
        draw_bbox = predict['bbox'][int(i)].copy()
        shrimp_bbox = predict['bbox'][int(i)].tolist()
        [p1,p2,p3,p4] = shrimp_bbox
        total_shrimp[str(i)] = {'bbox':shrimp_bbox,
                                'head':{},
                               'tail':{}}

        cv2.drawContours(clone, [draw_bbox], 0, color[0],2 )
        cv2.putText(clone,str(i),tuple(draw_bbox[3].tolist()) ,cv2.FONT_HERSHEY_SIMPLEX,1, 
                    color[pred_class], 2, cv2.LINE_AA)

        poly_path = mplPath.Path([p1,p2,p3,p4])

        
        

        for j in list(np.where(predict['classes']==1)[0]):
            head_bbox = predict['bbox'][int(j)].tolist()
            head_center =  (head_bbox[1][0]/2+head_bbox[3][0]/2,
                            head_bbox[1][1]/2+head_bbox[3][1]/2)
            if poly_path.contains_point(head_center) ==True :
                if is_not_in_other_shrimp2(head_center,judge_head) == True:
                    try:
                        judge_head.remove(shrimp_bbox)
                    except:
                        pass
                    draw_bbox = predict['bbox'][int(j)].copy()
                    cv2.drawContours(clone, [draw_bbox], 0, color[1],2 )
                    cv2.putText(clone,str(j),tuple(draw_bbox[3].tolist()) ,cv2.FONT_HERSHEY_SIMPLEX,1, 
                    color[1], 2, cv2.LINE_AA)
                    head = int(j)
                    total_shrimp[str(i)]['head'][head] = draw_bbox.tolist()
        


        for k in list(np.where(predict['classes']==2)[0]):
            tail_bbox = predict['bbox'][int(k)].tolist()
            tail_center =  (tail_bbox[1][0]/2+tail_bbox[3][0]/2,
                            tail_bbox[1][1]/2+tail_bbox[3][1]/2)
            if poly_path.contains_point(tail_center) == True  : 
                if is_not_in_other_shrimp2(tail_center,judge_tail) == True:
                    try:
                        judge_tail.remove(shrimp_bbox)
                    except:
                        pass
                    draw_bbox = predict['bbox'][int(k)].copy()            
                    cv2.drawContours(clone, [draw_bbox], 0, color[2],2 )
                    cv2.putText(clone,str(k),tuple(draw_bbox[3].tolist()) ,cv2.FONT_HERSHEY_SIMPLEX,1, 
                    color[2], 2, cv2.LINE_AA)
                    tail = int(k)
                    total_shrimp[str(i)]['tail'][tail] = draw_bbox.tolist()

   
    my_dict = total_shrimp.copy()

    
    return my_dict, clone
    
def caculate_lenght(matrix):
    all_distance = {}
    for i in range(len(matrix)):
        for j in range(i+1,len(matrix)):
            p1 = matrix[i]
            p2 = matrix[j]
            distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
            all_distance[str(i)+'-'+str(j)] = round(distance,2)
    # all_distance =dict(sorted(all_distance.items(), key=lambda item: item[1]))
    # short_side = list(all_distance.keys())[0:2]
    long_side = list(all_distance.keys())[2:4]
    lenght = (list(all_distance.values())[-3]+list(all_distance.values())[-4])/2
    return lenght,long_side


    
def caculate_width_2(long_sideID,shrimp_bbox):
    p1 = shrimp_bbox[int(long_sideID[0].split('-')[0])]
    p2 = shrimp_bbox[int(long_sideID[0].split('-')[1])]

    new_points1 = []
    for set_point in np.linspace(np.array(p1),np.array(p2),num = 4):
        new_points1.append(set_point.astype(int))
    new_points1 = sorted(new_points1,key = lambda x:x[0])



    p1 = shrimp_bbox[int(long_sideID[1].split('-')[0])]
    p2 = shrimp_bbox[int(long_sideID[1].split('-')[1])]

    new_points2 = []
    for set_point in np.linspace(np.array(p1),np.array(p2),num = 4):
        new_points2.append(set_point.astype(int))
    new_points2 = sorted(new_points2,key = lambda x:x[0])


    reset_point = []
    for i in range(len(new_points1)): 
        new_point = np.concatenate(([new_points1[i]],[new_points2[i]]),axis=0)
        new_point = sorted(new_point,key = lambda x:x[0])
        reset_point.append(new_point)
    return reset_point

def retengle(p1,p2,buffer):
    p1 = np.array(p1)
    p2 = np.array(p2)
    vector = p2-p1
    unit_vector = vector/np.linalg.norm(vector)

    a1 = buffer*np.array([-unit_vector[1],unit_vector[0]])

    new_points = [tuple(p1+a1),tuple(p1-a1),tuple(p2-a1),tuple(p2+a1)]
    points=[]
    for point in new_points:
        new_point = point[1],point[0]
        points.append(new_point)
    return points

def caculate_width_3(reset_point,mask_image,width_buffer):
    all_points = np.argwhere(mask_image == True)
    test_points = []
    length_pixel = []
    for point in reset_point:
        new_points = []
        new_points = retengle(point[0],point[1],width_buffer)        
        path = mplPath.Path(new_points)
        inside = path.contains_points(all_points)
        test_points.append(all_points[inside == True])
        length_pixel.append(len(inside[inside == True]))
    if length_pixel[1]>length_pixel[2]:
        width_index = 1
    else:
        width_index = 2
    width = length_pixel[width_index]/(width_buffer*2)
    return width, length_pixel, test_points, width_index
    
    
def caculate_width_length_area(data,pred_masks):
    shrimp_length_width = {}
    for shrimp_num in data:
        shrimp_bbox = data[str(shrimp_num)]['bbox']
        length, long_sideID = caculate_lenght(shrimp_bbox)


        mask_image = pred_masks[int(shrimp_num)]
        area = np.count_nonzero(mask_image[mask_image==True])

        reset_point = caculate_width_2(long_sideID,shrimp_bbox)

        width_buffer = 1
        width, length_pixel, test_points, width_index = caculate_width_3(reset_point,mask_image,width_buffer)

        shrimp_length_width[str(shrimp_num)] = {'length':round(length),'width':round(width),'area':area}
    return shrimp_length_width   
    
    
def statistic_shrimp(data):
    ID ,lengths, widths, areas = [], [], [], []
    for result in data:
        ID.append(result)
        lengths.append(data[result]['length'])
        widths.append(data[result]['width'])
        areas.append(data[result]['area'])
    df = pd.DataFrame({'ID':ID,'length':lengths,'width':widths,'area':areas})
    df = df.set_index('ID')
    df = df.describe()
    return df        
        
def predict_model(conf=0.7):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 
    cfg.MODEL.WEIGHTS = os.path.join('./output', "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor

