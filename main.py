import argparse
from ast import arg
import logging
from operator import itemgetter
import sys
import cv2
import numpy as np

DEFAULT_LOGGER_LEVEL = logging.INFO
LOGGER_NAME = 'Feature-Tracker'
LOGGER_FORMAT = '%(name)s %(levelname)s %(asctime)s:%(message)s'
logger = None


def setup_logger(level=DEFAULT_LOGGER_LEVEL, log_file=None):
    global logger
    logger = logging.getLogger (LOGGER_NAME)
    logger.setLevel(level)
    formatter = logging.Formatter(LOGGER_FORMAT)
    
    std_out_handler = logging.StreamHandler()
    std_out_handler.setLevel(level)
    std_out_handler.setFormatter(formatter)
    logger.addHandler(std_out_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def string_to_log_level(level_str):
    level_map = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'silent': logging.CRITICAL
        }
    try:
        return level_map[level_str]
    except KeyError:
        return logging.INFO

def load_data(img_path, video_path):
    img = cv2.imread(img_path) 
    if img is None:
        raise IOError(f'File {img_path} do not exist or is not a valid image format.')
        
    video_capture = cv2.VideoCapture(video_path) 
    if not video_capture.isOpened():
        raise IOError(f'File {video_path} do not exist or is not a valid video format.')

    return img, video_capture

    
def track_features(reference_image, video, outputDir, siftParam):
    size = (1280, 720)
    result = cv2.VideoWriter(outputDir, cv2.VideoWriter_fourcc(*'MJPG'), 20, size)
    paused = True 
    recieved, frame = video.read() 
    reference_image_bw = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    #sift = cv2.SIFT_create() 
    sift = cv2.xfeatures2d.SIFT_create() 
  
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    
    # FLANN Parameters 
    FLANN_INDEX_KDTREE = 1 
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = siftParam), 
    search_params = dict(checks=1) 
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image_bw, None)
    while recieved:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame_keypoints, frame_descriptors = sift.detectAndCompute(frame_bw, None)
        
        matches = matcher.match(reference_descriptors, frame_descriptors) 
        matches = sorted(matches, key = lambda x:x.distance)
        matches = [match.trainIdx for match in matches[:9]]

        matched_keypoints = [keypoint.pt for keypoint in itemgetter(*matches)(frame_keypoints)] 
        matched_keypoints = np.array(matched_keypoints, dtype=np.float32)

        bounding_rectangle = cv2.boundingRect(matched_keypoints)

        cv2.rectangle(frame, bounding_rectangle, [0,0,255]); 

        result.write(frame)

        cv2.imshow('Video', frame) 
        key = cv2.waitkey(2) 
        if key == ord('q'):

            break

        recieved, frame = video.read()
    
    video.release()
    result.release()

        
    cv2.destroyAllWindows()

def parse_arguments():
    parser = argparse. ArgumentParser (description=__doc__)
    parser.add_argument('--log_level', type=string_to_log_level, default=DEFAULT_LOGGER_LEVEL,
        help='Level at which logs will be saved')
    parser.add_argument('--log_file', type=str, default=None,
        help='File to which logs are saved, if none given logs will be displayed only on terminal')
    parser.add_argument('-i', '--image_path', type=str, required=True,
        help='Path to the image with object to track.')
    parser.add_argument('-v', '--video_path', type=str, required=False,
        help='Path to the video on witch object will be tracked.')
    parser.add_argument('-o', '--output', type=str, required=True,
        help='Path where output video will be saved.')
    parser.add_argument('-s', '--sift', type=str, required=False,
        help='Sift parameter value.')
    parser.add_argument('-f', '--fbm', type=str, required=False,
        help='FBM parameter value.')
    return parser.parse_args()
def main():
    args = parse_arguments()
    setup_logger (args.log_level, args.log_file)
    try:
        reference_image, video = load_data(args.image_path, args.video_path)

        if video is None:
            video = cv2.VideoCapture(0)

        if args.sift is None:
            siftParam = 5
        else:
            siftParam = args.sift
        track_features(reference_image, video, args.output, siftParam)
    except IOError as err:
        logger.critical(err.strerror)
        sys.exit(err.errno)




if __name__ == 'main':
    main()
else:
    logger = setup_logger()
    pass

