import numpy as np
import cv2
import torch
import plotly.graph_objects as go
from PIL import Image


def get_depth_estimator(device):
    """
    Initializes and returns a depth estimation pipeline.
    Uses 'depth-anything/Depth-Anything-V2-Small-hf' for fast & good results.
    Lazy import to avoid Python 3.12 compatibility issues with transformers/joblib.
    """
    from transformers import pipeline
    estimator = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device.index if device.type == "cuda" else -1
    )
    return estimator

def process_image_to_3d(estimator, image: Image.Image, downsample_factor=4):
    """
    Processes a single image to generate a 3D point cloud.
    """
    # 1. Get Depth Map
    depth_res = estimator(image)
    depth_map = depth_res['depth']  # PIL Image
    
    # 2. Convert to numpy
    depth_np = np.array(depth_map).astype(np.float32)
    img_np = np.array(image.convert("RGB"))
    
    # Optional: Resize both to be the same size if needed
    h, w = depth_np.shape
    img_np = cv2.resize(img_np, (w, h))

    # 3. Downsample for performance in Plotly
    depth_down = depth_np[::downsample_factor, ::downsample_factor]
    img_down = img_np[::downsample_factor, ::downsample_factor]
    
    hd, wd = depth_down.shape
    
    # Normalize depth map to avoid huge disparity
    if np.max(depth_down) > 0:
        depth_down = depth_down / np.max(depth_down)
        
    # Scale depth (invert it so objects are 'closer' to viewer)
    # Depth Anything outputs relative depth where larger values = closer
    z = depth_down * 1.0  

    # 4. Create coordinate grid
    x = np.linspace(-1, 1, wd)
    y = np.linspace(1, -1, hd) # Invert Y so up is up
    xx, yy = np.meshgrid(x, y)
    
    # Flatten
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    z_flat = z.flatten()
    
    colors = img_down.reshape(-1, 3) / 255.0
    
    return x_flat, y_flat, z_flat, colors

def generate_multi_view_point_cloud(images, device=torch.device("cpu"), downsample=4):
    """
    Takes multiple images and generates a single Plotly 3D Figure.
    In a simplistic version without complex Structure-from-Motion (SfM), 
    we extract the best view (or combine them loosely) and render the point cloud.
    Here we overlay the views linearly to give a multi-view 3D scatter effect.
    """
    estimator = get_depth_estimator(device)
    
    all_x, all_y, all_z = [], [], []
    all_colors = []
    
    # Just to create some separation in 3D space between views, 
    # we can translate them slightly based on view index.
    angle_step = 2 * np.pi / len(images)
    radius = 0.5
    
    for idx, img in enumerate(images):
        x, y, z, c = process_image_to_3d(estimator, img, downsample_factor=downsample)
        
        # We can add a simple heuristic rotation/translation to fake multi-view reconstruction
        # Real multi-view 3D requires SfM like COLMAP, but for Dashboard visualization,
        # scattering them with an angular orientation provides a visually stunning effect.
        angle = idx * angle_step
        
        # Rotate x, z
        x_rot = x * np.cos(angle) - z * np.sin(angle)
        z_rot = x * np.sin(angle) + z * np.cos(angle)
        
        # Translate to push out from center
        x_rot += radius * np.cos(angle)
        z_rot += radius * np.sin(angle)
        
        all_x.append(x_rot)
        all_y.append(y)
        all_z.append(z_rot)
        all_colors.append(c)
        
    x_comb = np.concatenate(all_x)
    y_comb = np.concatenate(all_y)
    z_comb = np.concatenate(all_z)
    colors_comb = np.concatenate(all_colors)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_comb, y=y_comb, z=z_comb,
        mode='markers',
        marker=dict(
            size=2,
            color=colors_comb,
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='lightgrey',
            camera=dict(
                eye=dict(x=0, y=1.0, z=-1.5),
                up=dict(x=0, y=1, z=0)
            )
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
