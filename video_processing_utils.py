import cv2
import open3d as o3d
import numpy as np
from open3d import geometry
import os


# Combine the original and mesh-rendered frames based on depth information
def combine_frames(original_frame, rendered_mesh_img, original_depth_frame, mesh_depth_buffer):
    # Create a mask where the mesh is closer than the original depth
    mesh_mask = mesh_depth_buffer < original_depth_frame
    
    # Initialize combined frames
    combined_frame = original_frame.copy()
    combined_depth = original_depth_frame.copy()
    
    # Update the combined frames with mesh information where the mask is True
    combined_frame[mesh_mask] = rendered_mesh_img[mesh_mask]
    combined_depth[mesh_mask] = mesh_depth_buffer[mesh_mask]
    
    return combined_frame, combined_depth

# Interpolate x value given two points and a y value
def interpolate_values(y, y0, y1, x0, x1):
    if y1 == y0:
        return x0
    return x0 + (x1 - x0) * (y - y0) / (y1 - y0)

# Compute barycentric coordinates for a point (x, y) with respect to a triangle
def compute_barycentric_coords(triangle, x, y):
    v0, v1, v2 = triangle
    detT = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    s = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / detT
    t = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / detT
    u = 1 - s - t
    return s, t, u

# Render a 3D mesh on a 2D image frame with depth buffering
def render_mesh_with_depth(mesh_vertices, vertex_colors, triangles, depth_frame, intrinsic):
    vertex_colors = np.asarray(vertex_colors)
    
    # Initialize depth and color buffers
    buffer_width, buffer_height = depth_frame.shape[1], depth_frame.shape[0]
    mesh_depth_buffer = np.ones((buffer_height, buffer_width)) * np.inf
    
    # Project 3D vertices to 2D image coordinates
    vertices_homogeneous = np.hstack((mesh_vertices, np.ones((mesh_vertices.shape[0], 1))))
    camera_coords = vertices_homogeneous.T[:-1,:]
    projected_vertices = intrinsic @ camera_coords
    projected_vertices /= projected_vertices[2, :]
    projected_vertices = projected_vertices[:2, :].T.astype(int)
    depths = camera_coords[2, :]

    mesh_color_buffer = np.zeros((buffer_height, buffer_width, 3), dtype=np.float32)
    
    # Loop through each triangle to render it
    for triangle in triangles:
        # Get 2D points and depths for the triangle vertices
        points_2d = np.array([projected_vertices[v] for v in triangle])
        triangle_depths = [depths[v] for v in triangle]
        colors = np.array([vertex_colors[v] for v in triangle])
        
        # Sort the vertices by their y-coordinates for scanline rendering
        order = np.argsort(points_2d[:, 1])
        points_2d = points_2d[order]
        triangle_depths = np.array(triangle_depths)[order]
        colors = colors[order]

        y_mid = points_2d[1, 1]

        for y in range(points_2d[0, 1], points_2d[2, 1] + 1):
            if y < 0 or y >= buffer_height:
                continue
            
            # Determine start and end x-coordinates for the current scanline
            if y < y_mid:
                x_start = interpolate_values(y, points_2d[0, 1], points_2d[1, 1], points_2d[0, 0], points_2d[1, 0])
                x_end = interpolate_values(y, points_2d[0, 1], points_2d[2, 1], points_2d[0, 0], points_2d[2, 0])
            else:
                x_start = interpolate_values(y, points_2d[1, 1], points_2d[2, 1], points_2d[1, 0], points_2d[2, 0])
                x_end = interpolate_values(y, points_2d[0, 1], points_2d[2, 1], points_2d[0, 0], points_2d[2, 0])
            
            x_start, x_end = int(x_start), int(x_end)

            # Loop through each pixel in the scanline
            for x in range(x_start, x_end + 1):
                if x < 0 or x >= buffer_width:
                    continue
                
                # Compute barycentric coordinates for the pixel
                s, t, u = compute_barycentric_coords(points_2d, x, y)
                
                # Check if the pixel lies inside the triangle
                if s >= 0 and t >= 0 and u >= 0:
                    # Interpolate depth and color for the pixel
                    depth_interp = s * triangle_depths[0] + t * triangle_depths[1] + u * triangle_depths[2]
                    color_interp = s * colors[0] + t * colors[1] + u * colors[2]
                    
                    # Update the pixel if it is closer to the camera
                    if depth_interp < mesh_depth_buffer[y, x]:
                        mesh_depth_buffer[y, x] = depth_interp
                        mesh_color_buffer[y, x] = color_interp

    # Convert float colors to uint8
    mesh_color_buffer = (mesh_color_buffer * 255).astype(np.uint8)
    
    return mesh_color_buffer, mesh_depth_buffer

# Create a video from a series of image frames
def render_video_from_frames(frame_directory, image_extension, save_directory, video_name, fps):
    # List all image files in the directory
    images = [img for img in os.listdir(frame_directory) if img.endswith(image_extension)]
    
    # Initialize video writer
    frame = cv2.imread(os.path.join(frame_directory, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))
    
    # Add each frame to the video
    for image in images:
        video.write(cv2.imread(os.path.join(frame_directory, image)))
        
    cv2.destroyAllWindows()


# Transform a mesh with scaling, translation, and rotation
def transform_mesh(mesh, scale, t_x, t_y, t_z, r_x, r_y, r_z):
    # Scale the mesh
    mesh.scale(scale, center=mesh.get_center())
    
    # Translate the mesh
    mesh = mesh.translate((t_x, t_y, t_z))
    
    # Rotate the mesh
    rotation_matrix = mesh.get_rotation_matrix_from_xyz((np.radians(r_x), np.radians(r_y), np.radians(r_z)))
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    
    return mesh