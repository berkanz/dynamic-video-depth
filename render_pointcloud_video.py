# %%
import numpy as np
import matplotlib.pyplot as plt
import open3d
from tqdm import tqdm
import argparse
import imageio
import matplotlib.pyplot as plt
from PIL import Image 
import open3d.visualization.rendering as rendering
import os
import cv2

TEST_RESULTS_PATH = 'ENTER YOUR BATCH PATH HERE'
file_names = [f for f in os.listdir(TEST_RESULTS_PATH) if f.endswith('.npz') and os.path.isfile(os.path.join(TEST_RESULTS_PATH, f))]
batch_count = len(file_names)
SAVE_DIRECTORY = "./pointclouds"
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

# %%

def convert_img_to_pointcloud(input_img, depth, intrinsics):
    y_size, x_size, _ = input_img.shape
    
    xx, yy = np.meshgrid(np.arange(x_size), np.arange(y_size))
    XYZ = np.zeros((y_size * x_size, 6))
    
    XYZ[:, 0] = (xx.flatten() - intrinsics[0, 2]) * depth.flatten() / intrinsics[0, 0]
    XYZ[:, 1] = (yy.flatten() - intrinsics[1, 2]) * depth.flatten() / intrinsics[1, 1]
    XYZ[:, 2] = depth.flatten()
    XYZ[:, 3:] = input_img.reshape(-1, 3)
    
    return XYZ

# %%

resolution = np.array((360,640))
intrinsics = np.array(([768.00, 0, 320.0], [0, 768.00, 180.0], [0.0, 0.0, 1.0]))
for i in tqdm(range(180, batch_count)):
    batch = np.load(os.path.join(TEST_RESULTS_PATH, file_names[i]))
    image = np.transpose(batch['img_1'], (2, 3, 1, 0))[:,:,:,0]    
    depth = np.transpose(batch['depth'], (2, 3, 1, 0))[:,:,0,0]
    image = cv2.resize(image, resolution, interpolation = cv2.INTER_AREA)
    depth = cv2.resize(depth, resolution, interpolation = cv2.INTER_AREA)
    pc = convert_img_to_pointcloud(image,depth,intrinsics)
    np.savetxt(f"pointclouds/batch_{i:03}.txt", pc)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc[:,:3])
    pcd.colors = open3d.utility.Vector3dVector(pc[:,3:])
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30,std_ratio=0.75)
    cleaned_pcd = pcd.select_by_index(ind)
    mtl = open3d.visualization.rendering.Material()
    mtl.shader = "defaultUnlit"
    render = rendering.OffscreenRenderer(resolution[1], resolution[0])
    center = np.array([0.03, -0.1, 1])  # look_at target
    eye = np.array([0.25, -0.3, 0.2])  # camera position
    up = np.array([0, -np.pi/2, 0])  # camera orientation
    vertical_field_of_view = 55
    aspect_ratio = resolution[0]/resolution[1] 
    near_plane = 0.5
    far_plane = 2
    fov_type = open3d.visualization.rendering.Camera.FovType.Vertical
    render.scene.add_geometry("generated_pointcloud", cleaned_pcd, mtl)
    render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)
    render.scene.camera.look_at(center, eye, up)
    img_o3d = np.array(render.render_to_image())
    fig=plt.figure()
    plt.ion()
    plt.imshow(img_o3d)
    plt.axis('off')
    plt.savefig(f"{SAVE_DIRECTORY}/pointcloud_render_{i:03}.png",bbox_inches='tight', dpi=129.075, pad_inches = 0)
    plt.close()
    
# %%
    
video_name = 'pointcloud_render.mp4'

images = [img for img in os.listdir(SAVE_DIRECTORY) if img.endswith(".png")]
frame = cv2.imread(os.path.join(SAVE_DIRECTORY, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 60, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(SAVE_DIRECTORY, image)))

cv2.destroyAllWindows()
video.release()