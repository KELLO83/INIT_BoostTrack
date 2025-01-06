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


def get_main_args():
    parser = make_parser()
    parser.add_argument("--data", type=str, default="data/cam0", help="Path to your custom dataset")  

    parser.add_argument("--no_reid", action="store_true", help="mark if visual embedding should NOT be used" , default=False)
    parser.add_argument("--no_cmc", action="store_true", help="mark if camera motion compensation should NOT be used" , default=False)
    parser.add_argument("--s_sim_corr", action="store_true", help="mark if you want to use corrected version of shape similarity calculation function")

    parser.add_argument("--btpp_arg_iou_boost", action="store_true", help="BoostTrack++ arg. Mark if only IoU should be used for detection confidence boost." , default=True)
    parser.add_argument("--btpp_arg_no_sb", action="store_true", help="BoostTrack++ arg. Mark if soft detection confidence boost should NOT be used."  , default= False )
    parser.add_argument("--btpp_arg_no_vt", action="store_true", help="BoostTrack++ arg. Mark if varying threhold should NOT be used for the detection confidence boost." , default=False)
    parser.add_argument('--video' , default = True)

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

    BoostTrackSettings.values['s_sim_corr'] = False

    BoostTrackPlusPlusSettings.values['use_rich_s'] = False
    BoostTrackPlusPlusSettings.values['use_sb'] = True
    BoostTrackPlusPlusSettings.values['use_vt'] = True


    detection_model = detector.Detector('external/weights/bytetrack_x_mot20.tar')  

    size = (640, 640)                                               
    # 모델 구조 상세 출력
    print("\n=== Model Structure ===")
    if detection_model.model is not None:
        print("\n2. Model Architecture:")
        if hasattr(detection_model.model, 'model'):
            print(detection_model.model.model)
        else:
            print(detection_model.model)
        print("\n3. Model Parameters:")
        total_params = sum(p.numel() for p in detection_model.model.parameters())
        print(f"Total parameters: {total_params:,}")
    else:
        print("Error: Model not initialized properly")


    
    loader = dataset.get_mot_loader(args.data, size=size)
    test = iter(loader)
    first_batch = next(test)
    
    # 데이터 구조 확인
    (img, np_img), label, info, idx = first_batch
    print("이미지 텐서 크기:", img.shape)
    print("NumPy 이미지 크기:", np_img.shape)
    print("라벨:", label)
    print("정보:", info)
    print("인덱스:", idx)

    # Test detection on first image
    print("\n=== Testing Object Detection ===")
    loader = dataset.get_mot_loader(args.data, size=size)
    
    loader_iter = iter(loader)
    frame_id = 0

    # for batch in loader_iter:
    #     (img, np_img), label, info, idx = batch
        
    #     display_img = np_img[0].numpy()
        
    #     if display_img.max() > 1.0:
    #         display_img = display_img.astype(np.uint8)
    #     else:
    #         display_img = (display_img * 255).astype(np.uint8)
        
    #     output = detection_model.forward(img.cuda())
    #     if output is not None:
    #         numpy_value = output.cpu().numpy()

    #         for det in numpy_value:
    #             x1, y1, x2, y2, score = det
                
    #             x1 = max(0, x1)
    #             y1 = max(0, y1)
                
    #             cv2.rectangle(display_img, 
    #                         (int(x1), int(y1)), 
    #                         (int(x2), int(y2)), 
    #                         (0, 255, 0), 2)
                
    #             score_text = f'{score:.2f}'
    #             cv2.putText(display_img, 
    #                     score_text, 
    #                     (int(x1), int(y1-10)), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 
    #                     0.9, (0, 0, 255), 2)

    #         cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
    #         cv2.imshow('Detections', display_img)
    #         cv2.waitKey(0)

            
    frame_count = 0
    total_time = 0
    frame_id = 0
    video_name = args.data
    tracker = None
    results = {video_name: []}

    loader_iter = iter(loader)
    for batch in loader_iter:
        (img, np_img), label, info, idx = batch
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

            # Initialize video writer
            if args.video:
                output_path = f"{video_name}_tracking.mp4"  # 현재 디렉토리에 저장
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                first_frame = cv2.imread(os.path.join(args.data, natsorted(os.listdir(args.data))[0]))
                frame_size = (first_frame.shape[1], first_frame.shape[0])
                out = cv2.VideoWriter(output_path, fourcc, 15.0, frame_size)
                print(f"\nVideo will be saved to: {os.path.abspath(output_path)}")

        output = detection_model.forward(img.cuda())
        if output is None:
            continue
        

        dets = output.cpu().numpy()
        
        start_time = time.time()

        targets = tracker.update(dets, img, np_img[0].numpy(), tag)
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])

        detection_img = display_img.copy()
        tracking_img = display_img.copy()
        
        detection_img = cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB)
        tracking_img = cv2.cvtColor(tracking_img, cv2.COLOR_BGR2RGB)

        for det in dets:
            x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(detection_img, f'Det:{score:.2f}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        

        for tlwh, track_id, conf in zip(tlwhs, ids, confs):
            color = get_id_color(track_id)
            x1, y1 = map(int, [tlwh[0], tlwh[1]])
            w, h = map(int, [tlwh[2], tlwh[3]])
            x2, y2 = x1 + w, y1 + h
       
            cv2.rectangle(tracking_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(tracking_img, f'ID:{track_id}', (x1, y1-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        # cv2.imshow('Detection', detection_img)
        # cv2.imshow('Tracking', tracking_img)
        # cv2.waitKey(0)

        if args.video:
            out.write(tracking_img)

  
        total_time += time.time() - start_time
        frame_count += 1
        
        

        results[video_name].append((frame_id, tlwhs, ids, confs))
        
    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    print(total_time)

    if args.video:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
