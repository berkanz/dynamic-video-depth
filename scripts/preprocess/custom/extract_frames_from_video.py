import cv2
import argparse
import os
DEFAULT_SAVE_PATH = '../../../datafiles/custom/JPEGImages/640p/custom/'
DEFAULT_RESIZE_FACTOR = 0.5

def cl_parser():
    parser = argparse.ArgumentParser(description="Video extraction I/O")
    parser.add_argument('--video_path', type=str, help='path to input video')
    parser.add_argument('--output_dir', default=DEFAULT_SAVE_PATH, type=str, help='save directory for extracted frames')
    parser.add_argument('--resize_factor', default=DEFAULT_RESIZE_FACTOR, type=float, help='factor of image resizing')
    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":
    
    args = cl_parser()
    vidcap = cv2.VideoCapture(args.video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, (0,0), fx=(args.resize_factor), fy=(args.resize_factor))
        cv2.imwrite(os.path.join(args.output_dir, f"frame{count:03}.jpg"), image)    
        success,image = vidcap.read()
        count += 1
    print(f"{count} frames extracted")