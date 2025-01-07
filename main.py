import os
from pickle import TRUE
import shutil
import time
import glob
import cv2
from cv2.gapi.ot import track
from natsort import natsorted

import dataset
import utils
from args import make_parser
from default_settings import GeneralSettings, get_detector_path_and_im_size, BoostTrackPlusPlusSettings, BoostTrackSettings
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack
import torch
import numpy as np
import random
from ultralytics import YOLO
"""
Script modified from Deep OC-SORT: 
https://github.com/GerardMaggiolino/Deep-OC-SORT
"""

id_colors = {}

def get_id_color(track_id):
    
    if track_id not in id_colors:
        id_colors[track_id] = [random.randint(150, 255) for _ in range(3)]

    color = id_colors[track_id]
    return color

def process_yolo_detection(results):
    "new id 사용"
    dets = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].cpu().numpy()
            cls = boxes.cls[i].cpu().numpy()
        
            if cls == 0:
                dets.append([x1, y1, x2, y2, conf])
                
    return np.array(dets) if dets else None


def get_main_args():
    parser = make_parser()
    parser.add_argument("--data", type=str, default="data/cam0", help="Path your dataset")  
    parser.add_argument('--reid_model', type=str, default='external/weights/mot17_sbs_S50.pth', help='Path to the reid model')
    
    parser.add_argument("--no_reid", action="store_true", help="mark if visual embedding should NOT be used" , default=False)
    parser.add_argument("--no_cmc", action="store_true", help="mark if camera motion compensation should NOT be used" , default=False)
    parser.add_argument("--s_sim_corr", action="store_true", help="mark if you want to use corrected version of shape similarity calculation function")

    parser.add_argument("--btpp_arg_iou_boost", action="store_true", help="BoostTrack++ arg. Mark if only IoU should be used for detection confidence boost." , default=True)
    parser.add_argument("--btpp_arg_no_sb", action="store_true", help="BoostTrack++ arg. Mark if soft detection confidence boost should NOT be used."  , default= False )
    parser.add_argument("--btpp_arg_no_vt", action="store_true", help="BoostTrack++ arg. Mark if varying threhold should NOT be used for the detection confidence boost." , default=False)
    parser.add_argument('--video' , default = False)


    args = parser.parse_args()
        
    for i in vars(args):
        print(f"{i}: {getattr(args, i)}")
        
    return args


def main():
    # Set dataset and detector
    args = get_main_args()
    GeneralSettings.values['dataset'] = 'custom'
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = True
    GeneralSettings.values['test_dataset'] = True
    GeneralSettings.values['reid_model'] = args.reid_model

    BoostTrackSettings.values['s_sim_corr'] = False

    BoostTrackPlusPlusSettings.values['use_rich_s'] = False
    BoostTrackPlusPlusSettings.values['use_sb'] = True
    BoostTrackPlusPlusSettings.values['use_vt'] = True


    #detection_model = detector.Detector('external/weights/bytetrack_x_mot20.tar')  
    detection_model = detector.Detector('external/weights/bytetrack_x_mot17.pth .tar')

    print("\n=== YOLOX Network  loading ... ===")
    if detection_model.model is not None:
        total_params = sum(p.numel() for p in detection_model.model.parameters())
        print(f"Total parameters: {total_params:,}")
    else:
        print("Error: Model not initialized")
        


    size = (640, 640)    
    loader = dataset.dataLoader(args.data, size=size) # args.data = 데이타 경로
    frame_id = 0       
    frame_count = 0
    total_time = 0
    video_name = args.data
    tracker = None
    
    #visualize_detections(loader , detection_model)

    for batch in loader:
        (img, np_img), _ , _ , _ = batch
        frame_id += 1
        
        tag = f"{video_name}:{frame_id}"
        img = img.cuda()
        

        display_img = np_img[0].numpy()
        if display_img.max() > 1.0:
            display_img = display_img.astype(np.uint8)
        else:
            display_img = (display_img * 255).astype(np.uint8)
        
        print(f"Processing {tag}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()

            tracker = BoostTrack(video_name=video_name)

            if args.video:
                string = args.reid_model.split('/')[-1].split('.')[0]
                print("string: ", string)
                output_path = f"{string}_out.mp4" 
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                first_frame = cv2.imread(os.path.join(args.data, natsorted(os.listdir(args.data))[0]))
                frame_size = (first_frame.shape[1], first_frame.shape[0])
                out = cv2.VideoWriter(output_path, fourcc, 15.0, frame_size)
                print(f"\nVideo will be saved to: {os.path.abspath(output_path)}")


    
        
        output = detection_model.forward(img.cuda())
        if output is None:
            continue
        
        output = output.cpu().numpy()
        

        dets = output.copy()
        
        start_time = time.time()

        targets = tracker.update(dets, img, np_img[0].numpy(), tag)
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])

        detection_img = display_img.copy()
        tracking_img = display_img.copy()
        
        detection_img = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
        tracking_img = cv2.cvtColor(tracking_img, cv2.COLOR_BGR2RGB)

        for idx , det in enumerate(dets):
            x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            color = get_id_color(idx)
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(detection_img, f'ID:{ idx }', (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for tlwh, track_id, conf in zip(tlwhs, ids, confs):
            color = get_id_color(track_id)
            x1, y1 = map(int, [tlwh[0], tlwh[1]])
            w, h = map(int, [tlwh[2], tlwh[3]])
            x2, y2 = x1 + w, y1 + h
       
            cv2.rectangle(tracking_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(tracking_img, f'ID:{track_id}', (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.imshow('Detection', detection_img)
        cv2.imshow('Tracking', tracking_img)
        cv2.waitKey(0)

        if args.video:
            out.write(tracking_img)

  
        total_time += time.time() - start_time
        frame_count += 1
            
    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    print(total_time)

    if args.video:
        out.release()

def visualize_detections(loader, detection_model):
    """
    mot17 mot 20 yolo 디버깅 확인용도
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('yolox_vision.mp4', fourcc, 15.0, (640, 640)) 
    
    for batch in loader:
        (img, np_img), _ , _ , _ = batch
        display_img = np_img[0].numpy()
        
        if display_img.max() > 1.0:
            display_img = display_img.astype(np.uint8)
        else:
            display_img = (display_img * 255).astype(np.uint8)
        
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        output = detection_model.forward(img.cuda())
        if output is not None:
            numpy_value = output.cpu().numpy()

            for det in numpy_value:
                x1, y1, x2, y2, score = det
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                
                cv2.rectangle(display_img, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                score_text = f'{score:.2f}'
                cv2.putText(display_img, 
                            score_text, 
                            (int(x1), int(y1-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 0, 255), 2)
    
        out.write(display_img)
    
    out.release()

if __name__ == "__main__":
    main()
