import tensorflow.compat.v1 as tf
import numpy as np
import argparse
import time
import json
import cv2
import os

from utils import annotator, change_channel, gray_normalizer, data_generator, smooth_contour, eye_aspect_ratio
from models import Simple, NASNET, Inception, GAP, YOLO
from hierarchical_dict import HierarchicalDict
from config import config
from logger import Logger
from tqdm import tqdm


tf.disable_v2_behavior()


def load_model(session, m_type, m_name, logger):
    # load the weights based on best loss
    best_dir = "best_loss"

    # check model dir
    model_path = "models/" + m_name
    path = os.path.join(model_path, best_dir)
    if not os.path.exists(path):
        raise FileNotFoundError

    if m_type == "simple":
        model = Simple(m_name, config, logger)
    elif m_type == "YOLO":
        model = YOLO(m_name, config, logger)
    elif m_type == "GAP":
        model = GAP(m_name, config, logger)
    elif m_type == "NAS":
        model = NASNET(m_name, config, logger)
    elif m_type == "INC":
        model = Inception(m_name, config, logger)
    else:
        raise ValueError

    # load the best saved weights
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        raise ValueError('There is no best model with given model')

    return model


def rescale(image):
    """
    If the input video is other than network size, it will resize the input video
    :param image: a frame form input video
    :return: scaled down frame
    """
    scale_side = max(image.shape)
    # image width and height are equal to 192
    scale_value = config["input_width"] / scale_side

    # scale down or up the input image
    scaled_image = cv2.resize(image, dsize=None, fx=scale_value, fy=scale_value)

    # convert to numpy array
    scaled_image = np.asarray(scaled_image, dtype=np.uint8)

    # one of pad should be zero
    w_pad = int((config["input_width"] - scaled_image.shape[1]) / 2)
    h_pad = int((config["input_width"] - scaled_image.shape[0]) / 2)

    # create a new image with size of: (config["image_width"], config["image_height"])
    new_image = np.ones((config["input_width"], config["input_height"]), dtype=np.uint8) * 250

    # put the scaled image in the middle of new image
    new_image[h_pad:h_pad + scaled_image.shape[0], w_pad:w_pad + scaled_image.shape[1]] = scaled_image

    return new_image


def upscale_preds(_preds, _shapes):
    """
    Get the predictions and upscale them to original size of video
    :param preds:
    :param shapes:
    :return: upscales x and y
    """
    # we need to calculate the pads to remove them from predicted labels
    pad_side = np.max(_shapes)
    # image width and height are equal to 384
    downscale_value = config["input_width"] / pad_side

    scaled_height = _shapes[0] * downscale_value
    scaled_width = _shapes[1] * downscale_value

    # one of pad should be zero
    w_pad = (config["input_width"] - scaled_width) / 2
    h_pad = (config["input_width"] - scaled_height) / 2

    # remove the pas from predicted label
    x = _preds[0] - w_pad
    y = _preds[1] - h_pad
    w = _preds[2]

    # calculate the upscale value
    upscale_value = pad_side / config["input_height"]

    # upscale preds
    x = x * upscale_value
    y = y * upscale_value
    w = w * upscale_value

    return x, y, w


def main_video(m_type, m_name, logger, video_path=None, write_output=True):
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)

        # check input source is a file or camera
        if video_path == None:
            video_path = 0

        # load the video or camera
        cap = cv2.VideoCapture(video_path)
        ret = True
        counter = 0
        tic = time.time()
        frames = []
        preds = []

        while ret:
            ret, frame = cap.read()

            if ret:
                # Our operations on the frame come here
                frames.append(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                f_shape = frame.shape
                if frame.shape[0] != 192:
                    frame = rescale(frame)

                image = gray_normalizer(frame)
                image = change_channel(image, config["input_channel"])
                [p] = model.predict(sess, [image])
                x, y, w = upscale_preds(p, f_shape)

                preds.append([x, y, w])
                # frames.append(gray)
                counter += 1

        toc = time.time()
        print("{0:0.2f} FPS".format(counter / (toc - tic)))

    # get the video size
    video_size = frames[0].shape[0:2]
    if write_output:
        # prepare a video write to show the result
        video = cv2.VideoWriter("predicted_video.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                (video_size[1], video_size[0]))

        for i, img in enumerate(frames):
            labeled_img = annotator((0, 250, 0), img, *preds[i])
            video.write(labeled_img)

        # close the video
        cv2.destroyAllWindows()
        video.release()
    print("Done...")


# load a the model with the best saved state from file and predict the pupil location
# on the input video. finaly save the video with the predicted pupil on disk
def main_images(m_type, m_name, logger, data_path=None, actors=[], write_output=True):
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    
    # this thing fixes something
    _config = ConfigProto()
    _config.gpu_options.allow_growth = True
    session = InteractiveSession(config=_config)
    # ---

    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)
        
        eye_info = HierarchicalDict(path=data_path + '/eye_data.json')

        for data_element in tqdm(data_generator(data_path=data_path, actors=actors), ascii=True):
            keys = data_element.keys
            image = data_element.item
            
            if not eye_info.check_key(keys):
                continue
            
            rois_coords = eye_info[keys]['roi']
            contours = eye_info[keys]['cnt']

            eye1 = np.array(contours[0]).reshape((-1, 1, 2))
            eye2 = np.array(contours[1]).reshape((-1, 1, 2))
            eyes = [eye1, eye2]

            # create empty mask to draw on
            result = np.zeros(image.shape, np.uint8)
            
            for i, (x1, x2, y1, y2) in enumerate(rois_coords):
                if eye_aspect_ratio(eyes[i]) < 0.2:
                    continue
                cv2.drawContours(result, [eyes[i]], 0, (255, 255, 255), -1) # fill the eye region with white color
                roi = image[y1 : y2, x1 : x2] # get the original eye region

                # preprocessing for the pupil detection model
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                shape = roi_gray.shape
                if roi_gray.shape[0] != 192:
                    roi_gray = rescale(roi_gray)
                roi_gray = gray_normalizer(roi_gray)
                roi_gray = change_channel(roi_gray, config["input_channel"])
                # ---

                [p] = model.predict(sess, [roi_gray])

                x, y, w = upscale_preds(p, shape)
                x, y, w = [int(item) for item in (x, y, w)]

                # draw the circle indicating a pupil
                roi = result[y1 : y2, x1 : x2]
                cv2.circle(roi, (x, y), 9, (0, 0, 255), -1)

                result[y1 : y2, x1 : x2] = roi
                # ---

            if write_output:
                actor, domain, segment, idx = keys
                path = os.path.dirname(os.path.abspath(f'{data_path}/{actor}/{domain}/{segment}'))
                path = path + '/' + segment + '/original_renders/eye_regions/'
                if not os.path.exists(path):
                    os.mkdir(path)
                cv2.imwrite(f'{path}{idx}.jpg', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    print("Done...")


if __name__ == "__main__":
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)

    parser.add_argument('--model_type',
                        help="INC, YOLO, simple",
                        default="INC")

    parser.add_argument('--model_name',
                        help="name of saved model (3A4Bh-Ref25)",
                        default="3A4Bh-Ref25")

    parser.add_argument('--video_path',
                        help="path to video file, empty for camera")

    parser.add_argument('--data_path',
                        help="path to a folder containing images, empty for video or camera")
    
    parser.add_argument('--actors', nargs='+', type=str,
                        help="list of actors to process")

    args = parser.parse_args()

    # model_name = args.model_name
    model_name = args.model_name
    model_type = args.model_type
    video_path = args.video_path
    data_path = args.data_path
    actors = args.actors or []


    # initial a logger
    logger = Logger(model_type, model_name, "", config, dir="models/")
    logger.log("Start inferring model...")

    # if video_path is not None:
    #     main_video(model_type, model_name, logger, video_path)
    
    if data_path is not None:
        main_images(model_type, model_name, logger, data_path, actors)
