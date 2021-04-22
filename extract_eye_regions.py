import argparse
import json
import cv2
import os

from landmark_detectors import FAN, MediapipeDetector, DlibDetector
from hierarchical_dict import HierarchicalDict
from face_alignment import LandmarksType
from roi import extract_eye_regions
from utils import data_generator
from skimage import io
from glob import glob
from tqdm import tqdm


def contours_to_int(contours):
    contours = [x.reshape((-1, 2)).tolist() for x in contours]
    for cnt in contours:
        for num, _ in enumerate(cnt):
            cnt[num][0] = int(cnt[num][0])
            cnt[num][1] = int(cnt[num][1])
    return contours


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

    data_dict = HierarchicalDict()

    for data_element in tqdm(data_generator(data_path, actors)):
        keys = data_element.keys
        image = data_element.item

        roi, contours = extract_eye_regions(landmark_detector, image)
        contours = contours_to_int(contours)

        data_dict[keys] = {
            'roi': roi,
            'cnt': contours
        }

    data_dict.save_json(data_path + '/data.json')
    print('Done.')
