import argparse
import json
import cv2
import os

from landmark_detectors import FAN, MediapipeDetector, DlibDetector, MtcnnDetector
from utils import data_generator, eye_aspect_ratio
from hierarchical_dict import HierarchicalDict
from face_alignment import LandmarksType
from roi import extract_eye_regions
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
    parser.add_argument("--data-path", required=True)
    parser.add_argument('--actors', nargs='+', type=str)
    parser.add_argument('--detector', default='FAN')

    args = parser.parse_args()


    if args.actors is None:
        actors = []
    else:
        actors = args.actors
    
    data_path = args.data_path


    detector_type = args.detector
    if detector_type == 'fan':
        landmark_detector = FAN(LandmarksType._2D, flip_input=False)
    elif detector_type == 'mediapioe':
        raise NotImplementedError
    elif detector_type == 'dlib':
        landmark_detector = DlibDetector()
    elif detector_type == 'mtcnn':
        raise NotImplementedError
        landmark_detector = MtcnnDetector()
    else:
        raise NotImplementedError

    if os.path.exists(data_path + '/eye_data.json'):
        json_path = data_path + '/eye_data.json'
    else:
        json_path = None

    data_dict = HierarchicalDict(json_path)
    segment_keys = []

    for data_element in tqdm(data_generator(data_path, actors)):
        keys = data_element.keys
        image = data_element.item

        result = extract_eye_regions(landmark_detector, image)
        if not result:
            continue

        roi, contours = result
        
        aspect_ratios = [eye_aspect_ratio(cnt) for cnt in contours]
        contours = contours_to_int(contours)

        if keys[:-1] not in segment_keys:
            segment_keys.append(keys[:-1])

        data_dict[keys] = {
            'roi': roi,
            'cnt': contours,
            'ars': aspect_ratios
        }

    data_dict.save_json(data_path + '/eye_data.json')
    print('Done.')
