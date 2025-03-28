import os
import re
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import glob


def create_video_from_images(
    image_folder: str,
    output_folder: str,
    output_filename: str = "output_video.mp4",
    fps: int = 30,
    file_pattern: str = "*.png",
    sort_numerically: bool = True,
    output_width: int = 720,
    output_height: int = 720,
    upscale_method: str = "nearest"
) -> str:
    """
    Create a high-quality video from a series of small grid images.

    Args:
        image_folder: Directory containing the image files
        output_folder: Directory where the output video will be saved
        output_filename: Name of the output video file
        fps: Frames per second for the output video
        file_pattern: Pattern to match image files
        sort_numerically: If True, sort files by embedded numbers
        output_width: Width of the output video (default: 720)
        output_height: Height of the output video (default: 720)
        upscale_method: Method to use for upscaling:
            - "nearest": Pixel-perfect scaling (sharp pixels)
            - "area": Better for downscaling
            - "bicubic": Smoother scaling
            - "lanczos": High-quality but slower scaling

    Returns:
        Path to the created video file
    """
    # Ensure directories exist
    image_folder = os.path.abspath(image_folder)
    output_folder = os.path.abspath(output_folder)
    
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of image files
    image_files = sorted(glob.glob(os.path.join(image_folder, file_pattern)))

    if not image_files:
        raise ValueError(f"No images found in {image_folder} matching pattern {file_pattern}")
    
    # Sort files numerically if requested
    if sort_numerically:
        def extract_number(filename):
            numbers = re.findall(r'\d+', os.path.basename(filename))
            if numbers:
                return int(numbers[-1])
            return 0
        
        image_files.sort(key=extract_number)
    
    # Define interpolation method for resizing
    interpolation = {
        'nearest': cv2.INTER_NEAREST,  # Sharpest for pixel art
        'area': cv2.INTER_AREA,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4  # Highest quality but slowest
    }.get(upscale_method, cv2.INTER_NEAREST)
    
    # Read the first image to get dimensions
    frame = cv2.imread(image_files[0])
    if frame is None:
        raise ValueError(f"Failed to read image: {image_files[0]}")
    
    # Set output video size
    size = (output_width, output_height)
    
    # For high quality, use H.264 codec with higher bitrate
    output_path = os.path.join(output_folder, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open. Check codec support.")

    print(f"Creating video from {len(image_files)} images...")
    print(f"Upscaling from {frame.shape[1]}x{frame.shape[0]} to {output_width}x{output_height} using {upscale_method} method")
    
    # Add each image to the video
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(image_files)}")
        
        frame = cv2.imread(image_file)
        if frame is not None:
            # Resize the image while maintaining pixel clarity
            resized_frame = cv2.resize(
                frame, 
                size, 
                interpolation=interpolation
            )
            
            # For very clean pixel art look, you can apply a sharpening kernel
            if upscale_method == "nearest":
                # Optional: Apply a slight sharpening filter to make edges crisper
                kernel = np.array([[-1, -1, -1], 
                                  [-1, 9, -1], 
                                  [-1, -1, -1]]) * 0.5 + np.array([[0, 0, 0], 
                                                                  [0, 1, 0], 
                                                                  [0, 0, 0]]) * 0.5
                resized_frame = cv2.filter2D(resized_frame, -1, kernel)
            
            out.write(resized_frame)
        else:
            print(f"Warning: Could not read image {image_file}")
    
    # Release the video writer
    out.release()
    print(f"Video saved to: {output_path}")
    
    # For even higher quality, you can re-encode the video with a higher bitrate
    # This requires having ffmpeg installed
    try:
        import subprocess
        # Create a higher quality version with ffmpeg
        high_quality_path = os.path.splitext(output_path)[0] + "_hq.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", output_path, 
            "-c:v", "libx264", "-crf", "18", # Lower CRF = higher quality (range 0-51)
            "-preset", "slow", # Slower preset = better compression
            high_quality_path
        ]
        subprocess.run(cmd, check=True)
        print(f"High quality video saved to: {high_quality_path}")
        return high_quality_path
    except Exception as e:
        print(f"Warning: Could not create high quality version: {e}")
        return output_path


def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename, assuming format like 'grid_obs_step_123.png'."""
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a high-quality video from small grid images.")
    parser.add_argument('--image_folder', type=str, required=True, help='Folder containing PNG images')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the MP4 video')
    parser.add_argument('--output_filename', type=str, default='output_video.mp4', help='Output video filename')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--pattern', type=str, default='*.png', help='File pattern to match')
    parser.add_argument('--width', type=int, default=720, help='Output video width')
    parser.add_argument('--height', type=int, default=720, help='Output video height')
    parser.add_argument('--upscale', type=str, default='nearest', 
                      choices=['nearest', 'area', 'bicubic', 'lanczos'],
                      help='Upscaling method (nearest=sharpest pixels)')
    
    args = parser.parse_args()
    
    create_video_from_images(
        args.image_folder, 
        args.output_folder, 
        args.output_filename,
        args.fps,
        args.pattern,
        output_width=args.width,
        output_height=args.height,
        upscale_method=args.upscale
    )