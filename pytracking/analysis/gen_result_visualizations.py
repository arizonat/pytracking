import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2 as cv
import torch
import numpy as np

sys.path.append('../..')
from pytracking.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist, Sequence
#from pytracking.analysis.playback_results import playback_result
from pytracking.evaluation.environment import env_settings

def get_plot_draw_styles():
    plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 0.0, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.0, 1.0), 'line_style': '-'},
                       {'color': (0.0, 1.0, 1.0), 'line_style': '-'},
                       {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                       {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                       {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                       {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                       {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                       {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                       {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                       {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                       {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                       {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]

    # From np to CV style colors
    for i in range(len(plot_draw_style)):
        color = plot_draw_style[i]['color']
        cv_color = [int(255*c) for c in color]
        plot_draw_style[i]['color'] = tuple(cv_color)

    return plot_draw_style

def gen_color_legend(square_size=10):
    settings = env_settings()
    result_plot_path = os.path.join(settings.result_plot_path, "color_palette.png")

    draw_styles = get_plot_draw_styles()
    colors = [c['color'] for c in draw_styles]

    im = 255*np.ones((square_size, square_size, 3)).astype('int')
    
    pass

def read_image(image_file: str):
    im = cv.imread(image_file)
    return cv.cvtColor(im, cv.COLOR_BGR2RGB)

def xywh_to_xyxy(xywh):
    x,y,w,h = xywh
    return [x,y,x+w,y+h]
    
def gen_and_save_visualized_results(trackers, dataset, output_name, include_frame_number=True, frame_num_loc=(25,25)):
    # Go through each sequence, make jpgs of all trackers on top of them

    plot_line_thickness = 2
    plot_draw_styles = get_plot_draw_styles()
    settings = env_settings()

    # Load the bbox results into memory
    print("Loading bbox results into memory...")
    bboxes = {}
    
    for seq_id, seq in enumerate(tqdm(dataset)):
        for trk_id, trk in enumerate(trackers):
            # Load results
            base_results_path = '{}/{}'.format(trk.results_dir, seq.name)
            results_path = '{}.txt'.format(base_results_path)

            if os.path.isfile(results_path):
                try:
                    pred_bbs = torch.tensor(np.loadtxt(str(results_path), dtype=np.float64))
                except:
                    pred_bbs = torch.tensor(np.loadtxt(str(results_path), delimiter=',', dtype=np.float64))
            else:
                raise Exception('Result not found. {}'.format(results_path))

            bboxes[(seq_id, trk_id)] = pred_bbs
            
    print("Generating and saving image...")
    for seq_id, seq in enumerate(tqdm(dataset)):
        tracker_results = []

        result_plot_path = os.path.join(settings.result_plot_path, output_name, seq.name)

        if not os.path.isdir(result_plot_path):
            os.makedirs(result_plot_path)
        
        # Go through each frame
        for frame_num, frame_path in enumerate(seq.frames):
            im = read_image(seq.frames[frame_num])
            
            # Go through each tracker
            for trk_id, trk in enumerate(trackers):
                pred_bb = bboxes[(seq_id, trk_id)]
                
                bb = pred_bb[frame_num]
                bb = [int(pt) for pt in bb]
                bb = xywh_to_xyxy(bb)
                im = cv.rectangle(im, (bb[0], bb[1]), (bb[2], bb[3]), plot_draw_styles[trk_id+1]['color'], plot_line_thickness)

            # Add the groundtruth on top
            gt_bb = seq.ground_truth_rect[frame_num]
            gt_bb = [int(pt) for pt in gt_bb]
            gt_bb = xywh_to_xyxy(gt_bb)

            im = cv.rectangle(im, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]), plot_draw_styles[0]['color'], plot_line_thickness)
            if include_frame_number:
                im = cv.putText(im, '#%05d'%(frame_num), frame_num_loc, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
            cv.imwrite(os.path.join(result_plot_path, '%05d.jpg'%(frame_num)),cv.cvtColor(im, cv.COLOR_RGB2BGR))

                
if __name__ == "__main__":
    # Sample usage
    trackers = []
    trackers.extend(trackerlist('dimp', 'super_dimp', range(0,1), 'DiMP'))
    dataset = get_dataset('fish')
    output_name = "Test"
    gen_and_save_visualized_results(trackers, dataset, output_name)
