import os
import sys
import numpy as np
import skimage.io
from PIL import Image
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath("./third_party/Mask_RCNN/")
# Directory of images to run detection on
IMAGE_DIR = './datafiles/custom/JPEGImages/640p/custom' 
# Directory to save your mask frames
SAVE_DIR = './datafiles/custom/Annotations/640p/custom' 

if __name__ == '__main__':

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from third_party.Mask_RCNN.mrcnn import utils
    import third_party.Mask_RCNN.mrcnn.model as modellib
    from third_party.Mask_RCNN.mrcnn import visualize
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.9

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']


    file_names = next(os.walk(IMAGE_DIR))[2]
    for index in tqdm(range(0, len(file_names))):
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[index]))
        # Run detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # In the next for loop, I check if extracted frame is larger than 16000 pixels, 
        # and if it is located minimum at 250th pixel in horizontal axis.
        # If not, I checked the next mask with "person" mask.
        current_mask_selection = 0
        while(True):
            if current_mask_selection<10:
                if (np.where(r["masks"][:,:,current_mask_selection]*1 == 1)[1].min()<250 or 
                    np.sum(r["masks"][:,:,current_mask_selection]*1)<16000):
                    current_mask_selection = current_mask_selection+1
                    continue
                elif (np.sum(r["masks"][:,:,current_mask_selection]*1)>16000 and 
                      np.where(r["masks"][:,:,current_mask_selection]*1 == 1)[1].min()>250):
                    break
            else:
                break
        mask = 255*(r["masks"][:,:,current_mask_selection]*1)
        mask_img = Image.fromarray(mask)
        mask_img = mask_img.convert('RGB')
        mask_img.save(os.path.join(SAVE_DIR, f"frame{index:03}.png"))











