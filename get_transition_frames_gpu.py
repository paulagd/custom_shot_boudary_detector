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


if __name__ == '__main__':
    # command line arguments --> file name, video_file_name, gpu or cpu
    '''
    - PARSE ARGS NORMAL
    - NO DELETE FRAMES IN ARG
    - REVISAR AMB UN VIDEO CURT SI DETECTA BÉ ELS SHOTS
    - MIRAR SI VOL QUE DETECTI QUANTS HA DETECAT O QUINA METRICA??
    - SAVE ALL FRAMES.pkl into a folder of pickles amb el nom del ID (nou pkl)
    '''
    s_time = time.time()
    # first decompose the video to frames
    # place the video to be detected into the directory

    video = sys.argv[1]   # 'name_file.mp4' # TODO: PANDAS SECTION OR NAME URL
    # pred_text_file_name = sys.argv[2]   # 'name_file_preds.txt' # TODO: PANDAS URL
    device = 'cuda'  # sys.argv[3]  # device 'cuda' or 'cpu'

    video_name = video.split('/')[-1].split('.')[0]
    pred_text_file_name = f'{video_name}.txt'

    print('decomposing video to frames this may take a while  for large videos :) .....')
    frames_path = f'video_frames/{video_name}'
    os.makedirs(frames_path, exist_ok=True)
    os.makedirs(os.path.join(frames_path, 'frames'), exist_ok=True)
    os.makedirs('predictions/', exist_ok=True)
    prediction_text_file = 'predictions/' + pred_text_file_name
    list_files = os.path.join(frames_path, f'{video_name}.txt')

    if not any(fname.endswith('.pickle') for fname in os.listdir(frames_path)):
        # do stuff if a file .pickle doesn't exist
        vid = VideoFileClip(video)
        vid = six_four_crop_video(vid)
        frames = [frame for i, frame in enumerate(vid.iter_frames()) if i % int(vid.fps)]
        with open(os.path.join(frames_path, f'frames_fps={vid.fps}.pickle'), 'wb') as f:
            pickle.dump(frames, f)
    else:
        pkl_fname = [fname for fname in os.listdir(frames_path) if fname.endswith('.pickle')][0]
        with open(os.path.join(frames_path, pkl_fname), 'rb') as f:
            frames = pickle.load(f)
    print('FRAMES GOTTEN!')

    if not os.path.exists(os.path.join(frames_path, 'frames', 'frame_1.png')):
        f = open(list_files, 'w+')

        for j, frame in enumerate(tqdm(frames, desc='saving frames to folder...')):
                frame_path = os.path.join(frames_path, 'frames', 'frame_' + str(j+1) + '.png')
                im = Image.fromarray(frame)
                im.save(frame_path)
                f.write(frame_path + '\n')
        f.close()
        print('frame decomposition complete !!! ')
    else:
        print('frame decomposition LOADED !!! ')

    # device = 'cuda'
    type_batch = 'torch.cuda.FloatTensor' if device == 'cuda' else 'torch.FloatTensor'


    #load model
    model = TransitionCNN()
    model.load_state_dict(torch.load('./models/shot_boundary_detector_even_distrib.pt'))
    model.to(device)

    pred_file = open(prediction_text_file, 'w+')

    loading_time = time.time() - s_time
    print('computing predictions for video', video, '...................' )

    test_video = TestVideo(list_files, sample_size=100, overlap=9)
    test_loader = DataLoader(test_video, batch_size=1, num_workers=16)

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
                        frame_index = video_indexes[indx][i+5]
                        pred_file.write(str(frame_index) + '\n')
    pred_file.close()

    # IDEA: delete files used for process
    # os.remove('frames.txt')
    # shutil.rmtree('video_frames/')
    shutil.rmtree(os.path.join(frames_path, 'frames'))

    print('Predictions complete !!!')
    print('Frames that are part of shot boundaries are listed in file the directory path predictions/' + pred_text_file_name)

    elapsed_time = time.time() - s_time

    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    h, r = divmod(loading_time, 3600)
    m, s = divmod(r, 60)

    print(f'LOADING TIME: {h:.2f} hours, {m:.2f} min, {s:.4f}seconds')
    print(f'TOTAL ELAPSED TIME: {hours:.2f} hours, {minutes:.2f} min, {seconds:.4f}seconds')
