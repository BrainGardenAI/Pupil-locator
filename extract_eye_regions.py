import argparse
import json
import cv2

from landmark_detectors import FAN, MediapipeDetector, DlibDetector
from face_alignment import LandmarksType
from roi import extract_eye_regions
from utils import data_generator
from skimage import io
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument('--actors', nargs='+', type=str)
    parser.add_argument('--detector', default='FAN')

    args = parser.parse_args()


    if args.actors is None:
        actors = []
    else:
        actors = args.actors
    
    data_path = args.input_folder


    detector_type = args.detector
    if detector_type == 'FAN':
        landmark_detector = FAN(LandmarksType._2D, flip_input=False)
    elif detector_type == 'Mediapipe':
        raise NotImplementedError
    elif detector_type == 'dlib':
        landmark_detector = DlibDetector()
    else:
        raise NotImplementedError


    eye_data = {}
    if not len(actors):
        actor_list = glob(data_path + '/*/')
    else:
        actor_list = [f'{data_path}/{actor}/' for actor in actors]

    data_dict = {}

    for actor_path in actor_list:
        actor_key = actor_path[:-1].split('/')[-1]
        data_dict[actor_key] = {}
        segment_list = glob(actor_path + 'real/*/')

        for segment_path in segment_list:
            segment_key = segment_path[:-1].split('/')[-1]
            data_dict[actor_key][segment_key] = {}
            frames = glob(segment_path + 'frames/*.jpg')

            for path in tqdm(frames, ascii=True):
                image = io.imread(path)
                idx = path.split('/')[-1][:-4]

                roi, contours = extract_eye_regions(landmark_detector, image)

                contours = [x.reshape((-1, 2)).tolist() for x in contours]
                for cnt in contours:
                    for num, _ in enumerate(cnt):
                        cnt[num][0] = int(cnt[num][0])
                        cnt[num][1] = int(cnt[num][1])
                
                data_dict[actor_key][segment_key][idx] = {
                    'roi': roi,
                    'cnt': contours
                }
    
    with open(data_path + '/data.json', 'w') as fp:
        json.dump(data_dict, fp)
