# %%
import open3d as o3d
from open3d import geometry
from video_processing_utils import *
from render_pointcloud_video import convert_img_to_pointcloud
from tqdm import tqdm

BATCH_DIRECTORY = #####
MESH_DIRECTORY = #####
SAVE_DIRECTORY = #####
RESOLUTION = np.array((720,480))
EXTRINSICS_PATH = #####
save_pointcloud = False
file_names = [f for f in os.listdir(BATCH_DIRECTORY) if f.endswith('.npz') and os.path.isfile(os.path.join(BATCH_DIRECTORY, f))]
batch_count = len(file_names)
intrinsics = np.array(([864.0, 0, 360], [0, 864.0, 240.0], [0.0, 0.0, 1.0]))
extrinsics_data = np.loadtxt(EXTRINSICS_PATH, skiprows=0, delimiter=" ")[:,1:]

# %%

mesh = o3d.io.read_triangle_mesh(MESH_DIRECTORY,print_progress = True)
translation_x, translation_y, translation_z = #####
rotation_x, rotation_y, rotation_z = #####
scale = #####


# %%

for i in tqdm(range(batch_count)):
    # batch means the batched information for a frame
    # including image itself, depth, flow estimation and more.
    batch = np.load(os.path.join(BATCH_DIRECTORY, file_names[i]))
    
    # loading the mesh
    mesh = o3d.io.read_triangle_mesh(MESH_DIRECTORY,print_progress = True)
    
    # pre-transformation of the mesh to the desired image section
    mesh =transform_mesh(mesh, scale, translation_x, 
                        translation_y, translation_z,
                        rotation_x, rotation_y, rotation_z) 
    
    # transformation of the mesh with the inverse camera extrinsics
    frame_transformation = np.vstack(np.split(extrinsics_data[i],4))
    inverse_frame_transformation = np.empty((4, 4))
    inverse_frame_transformation[:3, :] = np.concatenate((np.linalg.inv(frame_transformation[:3,:3]),  np.expand_dims(-1 * frame_transformation[:3,3],0).T), axis=-1)
    inverse_frame_transformation[3, :] = [0.00, 0.00, 0.00, 1.00]
    mesh.transform(inverse_frame_transformation)
    
    # resizing the image and depth to the desired resolution
    image = np.transpose(batch['img_1'], (2, 3, 1, 0))[:,:,:,0]    
    depth = np.transpose(batch['depth'], (2, 3, 1, 0))[:,:,0,0]
    image = cv2.resize(image, RESOLUTION, interpolation = cv2.INTER_AREA)
    image = (image * 255).astype(np.uint8)
    depth = cv2.resize(depth, RESOLUTION, interpolation = cv2.INTER_AREA)
    
    # rendering the color and depth buffer of the transformed mesh in the image domain
    mesh_color_buffer, mesh_depth_buffer = render_mesh_with_depth(np.array(mesh.vertices), np.array(mesh.vertex_colors), np.array(mesh.triangles), depth, intrinsics)
    
    # depth-aware overlaying of the mesh and the original image
    combined_frame, combined_depth = combine_frames(image, mesh_color_buffer, depth, mesh_depth_buffer) 
    combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
    
    # saving the overlayed image and point cloud
    cv2.imwrite(os.path.join(SAVE_DIRECTORY, f"frame_{i:03}.png"), combined_frame)
    if save_pointcloud:
        pointcloud = convert_img_to_pointcloud(combined_frame, combined_depth, intrinsics)
        np.savetxt(os.path.join(SAVE_DIRECTORY, f"batch_{i:03}.txt"), pointcloud)
    
# %%
    
video_name = 'depth_aware_object_insertion_demo.mp4'
save_directory = ""
frame_directory = SAVE_DIRECTORY
image_extension = ".png"
fps = 15 

# rendering a video of the overlayed frames
render_video_from_frames(frame_directory, image_extension, save_directory, video_name, fps)

# %%
