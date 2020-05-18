#!/usr/bin/env python3

import os, cv2, numpy as np

def load_images(images_path, as_array=True):
    """
    Load images from a folder. Return as ndarray.
    """
    file_list = os.listdir('./HBTN')
    images = []
    file_names = []
    for file in file_list:
        path = images_path + '/' + file
        with open(path, 'rb') as stream:
            image = cv2.imdecode(np.asarray(bytearray(stream.read()),
                                            dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        file_names.append(file)
    if as_array:
        images = np.asarray(images)
    return images, file_names
