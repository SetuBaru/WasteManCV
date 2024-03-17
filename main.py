import cv2
import torch
from PIL import Image
import os
import random
from datetime import datetime as dt

result_labels = {
    0: "Full",
    1: "Filling",
    2: "Empty"
}
model_path = 'yolov5/WasteManCV/weights/best.pt'


def detect(load_path, save_path=False, translate_labels=True, show=True):
    # Load Model
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')  # local model

    # Input Handler
    if os.path.isdir(load_path):
        print(f'Looking up in: {load_path}.\n')
        for file in os.listdir(load_path):
            print(f'Scanning {file}.\n')
            if str(file).endswith('.jpg') or str(file).endswith('.png'):
                # Check for escape key press
                if cv2.waitKey(1) == 27:
                    print("Escape key detected. Exiting...")
                    break
                # Images
                _path_ = os.path.join(load_path, file)
                _image_ = Image.open(_path_)  # PIL image
                results = model(_image_, size=416)  # batch of images
                if show is True:
                    results.show()  # Show Results
                # Check_Save_Path
                if save_path is True:
                    results.save(f'results/YOLO{random.randint(1, 99)}-{str(dt.now())}.jpg')
                elif save_path is False:
                    pass
                elif os.path.isdir(save_path):
                    results.save(f'{save_path}/YOLO{random.randint(1, 99)}-{str(dt.now())}.jpg')
                else:
                    print(f'Invalid Save Path: {save_path}, Please try again!\n')
                    return 'INVALSAVEPATH'

    elif os.path.isfile(load_path) and (str(load_path).endswith('.jpg') or str(load_path).endswith('.png')):
        _image_ = Image.open(load_path)  # PIL image
        # Inference
        results = model(_image_, size=416)  # batch of images
        # Results
        results.print()
        if show is True:
            results.show()
        if save_path is True:
            results.save(f'results/YOLO{random.randint(1, 99)}-{str(dt.now())}.jpg')
        elif save_path is False:
            pass
        elif os.path.isdir(save_path):
            results.save(f'{save_path}/YOLO{random.randint(1, 99)}-{str(dt.now())}.jpg')
        else:
            print(f'Invalid Save Path: {save_path}, Please try again!\n')
            return 'INVALSAVEPATH'

    else:
        print('Invalid File Path, Please try again!\n')
        return 'INVALPATH'


if __name__ == '__main__':
    target1 = 'test_data'
    target2 = 'test2'
    target3 = 'test2/2_jpg.rf.a1a4e758d2b90374c4a930615a268b4b.jpg'
    # detect(target3) # Single Image Test
    detect(target3)  # Batch Image Test
