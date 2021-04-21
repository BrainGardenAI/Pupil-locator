import mediapipe as mp
import numpy as np
import dlib
import cv2

from face_alignment import FaceAlignment
from collections import OrderedDict
from imutils import face_utils


class FAN(FaceAlignment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_eyes(self, image):
        prediction = self.get_landmarks(image)
        eye1 = prediction[0][36 : 42]
        eye2 = prediction[0][42 : 48]

        return eye1, eye2


class MediapipeDetector:
    def __init__(self):
        self.model = mp.solutions.face_mesh
    
    def get_eyes(self, image):
        with self.model.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(image)
            # TODO: extract eye landmarks and transform to image pixel coordinates
            return results


class DlibDetector:
    _landmark_to_idx = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 35)),
        ("jaw", (0, 17))
    ])

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor.dat')

    def get_eyes(self, image):
        face_rect = self.detector(image)
        assert len(face_rect) != 0
        landmarks = self.predictor(image, face_rect[0])

        right_eye_indices = self._landmark_to_idx['right_eye']
        left_eye_indices = self._landmark_to_idx['left_eye']
        
        right_eye = self._get_part(landmarks, right_eye_indices)
        left_eye = self._get_part(landmarks, left_eye_indices)
        
        right_eye = np.array(right_eye).reshape(-1, 1, 2)
        left_eye = np.array(left_eye).reshape(-1, 1, 2)
        return right_eye, left_eye
    
    def _get_part(self, landmarks, indices):
        landmarks_coords = []
        for idx in range(*indices):
            x = landmarks.part(idx).x
            y = landmarks.part(idx).y 
            landmarks_coords.append([x, y])
        return landmarks_coords