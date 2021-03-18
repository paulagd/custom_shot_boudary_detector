import sys, time, os, torch, shutil, pickle
from moviepy.editor import VideoFileClip
from video_processing import six_four_crop_video
from utilities import find_all_urls
from tqdm import tqdm
from PIL import Image

from IPython import embed


def save_in_disk(video_id, frames, save_path, frames_folder='frames_64'):
    list_files = os.path.join(save_path, f"{frames_folder.split('_')[-1]}_list_{video_id}.txt")
    os.makedirs(os.path.join(save_path, frames_folder), exist_ok=True)
    f = open(list_files, 'w+')
    for j, frame in enumerate(tqdm(frames, desc='saving frames to folder...')):
        frame_path = os.path.join(save_path, frames_folder, 'frame_' + str(j + 1) + '.png')
        im = Image.fromarray(frame)
        im.save(frame_path)
        f.write(frame_path + '\n')
    f.close()
    print(f"frames decomposition complete in DISK {frames_folder.split('_')[-1]} !!!")

    
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
    main_file_path = f'continguts_indexacio.pkl'
    save_path = f'videos/sample_2000_videos'
    os.makedirs(save_path, exist_ok=True)

    #
    # main_path_urls = os.path.join(args.dataset_path, args.file_name)
    # url_list, class_name, durada = find_all_urls(main_path_urls, tv3=not args.demo_videos)
    df = find_all_urls(main_file_path)
    df = df[37:]
    TOTAL_DURATION = 0.0
    for i, row in tqdm(df.iterrows(), desc='Loading videos and saving frames to DISK...', total=len(df)):
        serie_time = time.time()
        save_custom_path = os.path.join(save_path, str(int(row['content_id'])))
        video_name = f"{row['promo']}_{int(row['content_id'])}"
        if not os.path.exists(save_custom_path):
            os.makedirs(save_custom_path, exist_ok=True)
            vid = VideoFileClip(row['url'])
            if not os.path.exists(os.path.join(save_custom_path, f"frames_64", 'frame_1.png')):
                vid_6464 = six_four_crop_video(vid)
                frames_6464 = [frame for frame in vid_6464.iter_frames()]
                # TODO: save in disk 64x64:
                save_in_disk(video_name, frames_6464, save_custom_path, frames_folder='frames_64')
                vid_6464.close()
            # TODO: save in disk 1 every second:
            if not os.path.exists(os.path.join(save_custom_path, f"frames_original_{vid.size[0]}x{vid.size[1]}", 'frame_1.png')):
                # frames = [frame for i, frame in enumerate(vid.iter_frames()) if i % int(vid.fps)]
                frames = [vid.get_frame(t=i) for i in range(int(vid.duration))]

                # vid.iter_frames()[::int(vid.fps)]
                # vid.reader.nframes
                # vid_6464.reader.nframes
                save_in_disk(video_name, frames, save_custom_path, frames_folder=f"frames_original_{vid.size[0]}x{vid.size[1]}")
            vid.close()
            hours, rem = divmod(time.time() - serie_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f'It took {hours:.2f}h, {minutes:.2f}min, {seconds:.4f}sec to load a video of duration ' +
                  time.strftime('%H:%M:%S', time.gmtime(vid.duration)))
            TOTAL_DURATION += vid.duration

    elapsed_time = time.time() - s_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f'TOTAL ELAPSED TIME:\n')
    print(f'It took {hours:.2f}h, {minutes:.2f}min, {seconds:.4f}sec to process several VIDEOS of TOTAL duration == '
          + time.strftime('%H:%M:%S', time.gmtime(TOTAL_DURATION)))

