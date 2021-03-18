import sys, time, os, torch, shutil, pickle

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
from snippet import getSnippet
from math import floor
from transition_network import TransitionCNN
from utilities import normalize_frame, print_shape
import pandas as pd
from TestVideo import TestVideo, return_start_and_end
from moviepy.editor import VideoFileClip
from video_processing import six_four_crop_video
from PIL import Image
from IPython import embed
from tqdm import tqdm


def do_inference(list_files, id_video, device, num_workers=0):
    # device = 'cuda'
    inference_time = time.time()
    pred_file = open(os.path.join('predictions', f'{id_video}.txt'), 'w+')
    type_batch = 'torch.cuda.FloatTensor' if device == 'cuda' else 'torch.FloatTensor'
    # load model
    model = TransitionCNN()
    model.load_state_dict(torch.load('./models/shot_boundary_detector_even_distrib.pt'))
    model.to(device)
    test_video = TestVideo(list_files, sample_size=100, overlap=9)
    test_loader = DataLoader(test_video, batch_size=1, num_workers=num_workers)

    video_indexes = []
    vals = np.arange(test_video.get_line_number())
    length = len(test_video)

    for val in range(length):
        s, e = return_start_and_end(val)
        video_indexes.append(vals[s:e])

    for indx, batch in enumerate(test_loader):
        batch.to(device)
        batch = batch.type(type_batch)
        predictions = model(batch)
        predictions = predictions.argmax(dim=1).cpu().numpy()
        for idx, prediction_set in enumerate(predictions):
            for i, prediction in enumerate(prediction_set):
                if prediction[0][0] == 0:
                    frame_index = video_indexes[indx][i + 5]
                    pred_file.write(str(frame_index) + '\n')
    pred_file.close()
    res = inference_time - time.time()
    hours, rem = divmod(res, 3600)
    m, s = divmod(rem, 60)
    print('> computed predictions for video', id_video, f'in {int(s):02d} seconds')


if __name__ == '__main__':
    # command line arguments --> file name, video_file_name, gpu or cpu
    '''
    - GET SHOTS
    '''
    s_time = time.time()
    device = 'cuda'  # sys.argv[3]  # device 'cuda' or 'cpu'
    videos_path = f'videos/sample_2000_videos'
    os.makedirs('predictions/', exist_ok=True)

    for id, id_video in enumerate(os.listdir(videos_path)):
        try:
            file = [file for file in os.listdir(os.path.join(videos_path, id_video)) if file.startswith('64_list')][0]
            path_list_files_64 = os.path.join(videos_path, id_video, file)
            if not os.path.exists(os.path.join('predictions', f'{id_video}.txt')):
                do_inference(path_list_files_64, id_video, device)
            else:
                print('> Predictions for video', id_video, f'already DONE!')
        except:
            print(f'Stopped at iteration {id} | video id = {id_video}') # 40
            break

    print('Predictions complete !!!')
    print(f'Frames that are part of shot boundaries are listed in file the directory path predictions/{id_video}.txt')

    elapsed_time = time.time() - s_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'TOTAL ELAPSED TIME: {hours:.2f} hours, {minutes:.2f} min, {seconds:.4f}seconds')
