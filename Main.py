#!/usr/bin/env python3
# Complete Ultra-High-Resolution Image Stitcher with Gap Filling
# CSCI 576 Multimedia Project - Spring 2025
#High processing time version

import cv2
import numpy as np
import os
import argparse
import time
import logging
from tqdm import tqdm
import sys
import gc
from skimage import transform as tf
from scipy import ndimage
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class CompletePanoramaStitcher:
    def __init__(self, output_dir="output", temp_dir="temp_frames", 
                 max_resolution=2000, debug=False):
        """Initialize the complete panorama stitcher with gap filling capabilities"""
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.max_resolution = max_resolution
        self.debug = debug
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Initialize feature detector
        self.detector = cv2.SIFT_create(nfeatures=3000)
        
        # Initialize feature matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_frames(self, video_path, sample_rate=3):
        """Extract frames from a video file at the specified sample rate"""
        logger.info(f"Extracting frames from {video_path} with sample rate {sample_rate}")
        
        # Generate unique prefix for this video's frames
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return []
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video info: {total_frames} frames, {fps} FPS")
        
        # Calculate expected frames to extract
        expected_frames = total_frames // sample_rate
        logger.info(f"Will extract approximately {expected_frames} frames")
        
        # Extract frames with progress bar
        frames = []
        frame_count = 0
        saved_count = 0
        
        pbar = tqdm(total=expected_frames, desc=f"Extracting frames from {video_name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified interval
            if frame_count % sample_rate == 0:
                # Save frame to disk
                frame_path = os.path.join(self.temp_dir, f"{video_name}_{saved_count:04d}.jpg")
                
                # Use high quality JPEG compression for less loss
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames.append(frame_path)
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1
            
            # Force garbage collection periodically
            if frame_count % 100 == 0:
                gc.collect()
        
        pbar.close()
        cap.release()
        logger.info(f"Extracted {saved_count} frames from {video_path}")
        
        return frames

    def find_matching_features(self, img1, img2):
        """Find matching features between two images"""
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        # Match descriptors
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, None, 0
        
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                if m.queryIdx < len(kp1) and m.trainIdx < len(kp2):
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return None, None, 0
        
        # Extract coordinates of matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matched_count = np.sum(mask)
        
        return H, mask, matched_count

    def build_image_graph(self, image_paths):
        """Build a graph of images based on feature matches between pairs"""
        logger.info("Building image relationship graph...")
        
        n_images = len(image_paths)
        graph = {i: [] for i in range(n_images)}
        homographies = {}
        
        # Load all images to memory (can be optimized further if memory is a constraint)
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                # Resize large images for feature detection
                h, w = img.shape[:2]
                if max(h, w) > self.max_resolution:
                    scale = self.max_resolution / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                images.append(img)
            else:
                logger.warning(f"Could not read image: {path}")
                images.append(None)
        
        # Find matches between all pairs of images
        pbar = tqdm(total=n_images * (n_images - 1) // 2, desc="Finding matches")
        for i in range(n_images):
            if images[i] is None:
                continue
                
            for j in range(i+1, n_images):
                if images[j] is None:
                    continue
                    
                # Find homography between images
                H, mask, matched_count = self.find_matching_features(images[i], images[j])
                
                if H is not None and matched_count >= 10:
                    # Store edge in graph
                    graph[i].append((j, matched_count))
                    graph[j].append((i, matched_count))
                    
                    # Store homography
                    homographies[(i, j)] = H
                    homographies[(j, i)] = np.linalg.inv(H)
                
                pbar.update(1)
        
        pbar.close()
        
        return graph, homographies, images

    def find_optimal_spanning_tree(self, graph):
        """Find an optimal spanning tree from the image graph using Prim's algorithm"""
        logger.info("Finding optimal stitching order...")
        
        if not graph:
            return []
            
        # Find node with most connections as starting point
        start_node = max(graph.keys(), key=lambda x: sum(count for _, count in graph[x]) if x in graph else 0)
        
        # Initialize spanning tree
        tree_nodes = {start_node}
        tree_edges = []
        
        # Implement Prim's algorithm
        while len(tree_nodes) < len(graph):
            best_edge = None
            best_weight = -1
            
            # Check all edges from tree to non-tree nodes
            for node in tree_nodes:
                for neighbor, weight in graph[node]:
                    if neighbor not in tree_nodes and weight > best_weight:
                        best_edge = (node, neighbor)
                        best_weight = weight
            
            if best_edge is None:
                # No more edges to add
                break
                
            # Add the best edge
            tree_edges.append(best_edge)
            tree_nodes.add(best_edge[1])
        
        return tree_edges

    def compute_global_transforms(self, spanning_tree, homographies, n_images):
        """Compute global transformation matrices for all images based on the spanning tree"""
        logger.info("Computing global transformations...")
        
        # Initialize global transforms with identity for all images
        global_transforms = {i: np.eye(3) for i in range(n_images)}
        
        # Create an adjacency list representation of the spanning tree
        tree_adj = {i: [] for i in range(n_images)}
        for src, dst in spanning_tree:
            tree_adj[src].append(dst)
            tree_adj[dst].append(src)
        
        # Choose a root node (first node in the tree)
        root = spanning_tree[0][0]
        
        # Perform breadth-first search to compute global transforms
        queue = [root]
        visited = {root}
        
        while queue:
            node = queue.pop(0)
            
            for neighbor in tree_adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    # Compute relative transform
                    if (node, neighbor) in homographies:
                        H_relative = homographies[(node, neighbor)]
                        global_transforms[neighbor] = global_transforms[node] @ H_relative
                    else:
                        logger.warning(f"Missing homography for edge ({node}, {neighbor})")
        
        return global_transforms

    def estimate_panorama_size(self, global_transforms, images):
        """Estimate the size of the final panorama and adjust transformations"""
        logger.info("Estimating panorama dimensions...")
        
        # Collect all corner points
        all_corners = []
        for idx, img in enumerate(images):
            if img is None or idx not in global_transforms:
                continue
                
            h, w = img.shape[:2]
            corners = np.array([
                [0, 0],
                [0, h-1],
                [w-1, h-1],
                [w-1, 0]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform corners
            warped_corners = cv2.perspectiveTransform(corners, global_transforms[idx])
            all_corners.append(warped_corners)
        
        # Combine all corners and find bounding box
        all_corners = np.concatenate(all_corners, axis=0)
        min_x, min_y = np.int32(np.floor(all_corners.min(axis=0)[0]) - 0.5)
        max_x, max_y = np.int32(np.ceil(all_corners.max(axis=0)[0]) + 0.5)
        
        # Calculate translation
        translation_x = -min_x if min_x < 0 else 0
        translation_y = -min_y if min_y < 0 else 0
        
        # Create translation matrix
        translation_matrix = np.array([
            [1, 0, translation_x],
            [0, 1, translation_y],
            [0, 0, 1]
        ])
        
        # Adjust all transformations
        for idx in global_transforms:
            global_transforms[idx] = translation_matrix @ global_transforms[idx]
        
        # Calculate dimensions
        width = max_x - min_x + 2 * translation_x
        height = max_y - min_y + 2 * translation_y
        
        # Cap dimensions if necessary
        max_allowed_dim = 15000
        if width > max_allowed_dim or height > max_allowed_dim:
            logger.warning(f"Panorama size exceeds limit: {width}x{height}")
            scale = max_allowed_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            
            # Adjust transformations for scaling
            scale_matrix = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1]
            ])
            
            for idx in global_transforms:
                global_transforms[idx] = scale_matrix @ global_transforms[idx]
        
        logger.info(f"Estimated panorama dimensions: {width}x{height}")
        return width, height, global_transforms

    def create_initial_panorama(self, images, global_transforms, width, height):
        """Create an initial panorama with potential gaps"""
        logger.info("Creating initial panorama...")
        
        # Create empty panorama
        panorama = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Warp and blend each image
        for idx, img in enumerate(images):
            if img is None or idx not in global_transforms:
                continue
                
            # Warp image
            warped = cv2.warpPerspective(img, global_transforms[idx], (width, height))
            
            # Create mask for this image
            img_mask = np.any(warped > 0, axis=2).astype(np.uint8) * 255
            
            # Blend with existing panorama using feathering
            panorama_alpha = mask / 255.0
            img_alpha = img_mask / 255.0
            
            # Calculate combined alpha
            combined_alpha = panorama_alpha + img_alpha - panorama_alpha * img_alpha
            combined_alpha = np.clip(combined_alpha, 1e-10, 1.0)  # Avoid division by zero
            
            # Calculate new panorama
            for c in range(3):
                panorama[:,:,c] = (
                    (1 - img_alpha) * panorama_alpha * panorama[:,:,c] + 
                    img_alpha * warped[:,:,c]
                ) / combined_alpha
            
            # Update mask
            mask = np.maximum(mask, img_mask)
        
        return panorama, mask

    def fill_panorama_gaps(self, panorama, mask):
        """Fill gaps in the panorama using various techniques"""
        logger.info("Filling gaps in the panorama...")
        
        # Create a binary mask for holes
        holes = (mask == 0).astype(np.uint8)
        
        # If no holes, return the original panorama
        if not np.any(holes):
            logger.info("No gaps found in the panorama")
            return panorama
        
        # Identify contiguous hole regions
        num_labels, labels = cv2.connectedComponents(holes)
        logger.info(f"Found {num_labels-1} separate gap regions to fill")
        
        # Fill each hole region
        filled_panorama = panorama.copy()
        
        for label in range(1, num_labels):
            # Extract the current hole
            current_hole = (labels == label).astype(np.uint8)
            
            # Find the boundary pixels around this hole
            kernel = np.ones((3, 3), np.uint8)
            boundary = cv2.dilate(current_hole, kernel) - current_hole
            
            # If boundary is empty, skip this hole
            if not np.any(boundary):
                continue
            
            # Get coordinates of boundary pixels
            boundary_y, boundary_x = np.where(boundary > 0)
            
            # Get values of boundary pixels
            boundary_values = panorama[boundary_y, boundary_x]
            
            # Find coordinates of hole pixels
            hole_y, hole_x = np.where(current_hole > 0)
            
            # For small holes, use simple averaging
            if len(hole_y) < 500:
                # Use inpainting
                filled_region = cv2.inpaint(
                    panorama, 
                    current_hole, 
                    3, 
                    cv2.INPAINT_TELEA
                )
                filled_panorama = np.where(
                    np.expand_dims(current_hole, axis=2).astype(bool),
                    filled_region,
                    filled_panorama
                )
            else:
                # For larger holes, use more advanced techniques
                # Calculate the mean color from the boundary
                mean_color = np.mean(boundary_values, axis=0)
                
                # Use the mean color to fill the hole
                for i in range(len(hole_y)):
                    filled_panorama[hole_y[i], hole_x[i]] = mean_color
                
                # Apply Gaussian blur to the filled region for better blending
                blurred = cv2.GaussianBlur(filled_panorama, (15, 15), 0)
                filled_panorama = np.where(
                    np.expand_dims(current_hole, axis=2).astype(bool),
                    blurred,
                    filled_panorama
                )
        
        return filled_panorama

    def enhance_panorama_quality(self, panorama):
        """Apply various enhancements to improve the final panorama quality"""
        logger.info("Enhancing panorama quality...")
        
        try:
            # Convert to LAB color space for better color adjustments
            lab = cv2.cvtColor(panorama, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to the L channel for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Add slight sharpening
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
            
            return sharpened
        except Exception as e:
            logger.error(f"Error enhancing panorama: {e}")
            return panorama

    def create_complete_panorama(self, image_paths):
        """Create a complete panorama from multiple images with gap filling"""
        logger.info(f"Starting panorama creation with {len(image_paths)} images")
        
        # Build image relationship graph
        graph, homographies, images = self.build_image_graph(image_paths)
        
        # Find optimal spanning tree
        spanning_tree = self.find_optimal_spanning_tree(graph)
        
        if not spanning_tree:
            logger.error("Could not create a valid stitching order")
            return None
        
        # Compute global transformations
        global_transforms = self.compute_global_transforms(spanning_tree, homographies, len(images))
        
        # Estimate panorama size
        width, height, adjusted_transforms = self.estimate_panorama_size(global_transforms, images)
        
        # Create initial panorama
        panorama, mask = self.create_initial_panorama(images, adjusted_transforms, width, height)
        
        if self.debug:
            # Save the initial panorama with gaps
            initial_path = os.path.join(self.output_dir, "initial_panorama.jpg")
            cv2.imwrite(initial_path, panorama)
            
            # Save the mask
            mask_path = os.path.join(self.output_dir, "panorama_mask.jpg")
            cv2.imwrite(mask_path, mask)
        
        # Fill gaps in the panorama
        filled_panorama = self.fill_panorama_gaps(panorama, mask)
        
        # Enhance the panorama quality
        final_panorama = self.enhance_panorama_quality(filled_panorama)
        
        return final_panorama

    def process_videos(self, video_paths, output_path="panorama.jpg", sample_rate=3):
        """Process multiple videos to create a complete panorama"""
        start_time = time.time()
        
        # Extract frames from all videos
        all_frames = []
        for video_path in video_paths:
            frames = self.extract_frames(video_path, sample_rate)
            all_frames.extend(frames)
        
        if not all_frames:
            logger.error("No frames were extracted from videos")
            return None
        
        # Create panorama
        panorama = self.create_complete_panorama(all_frames)
        
        if panorama is None:
            logger.error("Failed to create panorama")
            return None
        
        # Save the final panorama
        logger.info(f"Saving final panorama to {output_path}")
        cv2.imwrite(output_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Create a preview version
        h, w = panorama.shape[:2]
        max_preview_dim = 1500
        if max(h, w) > max_preview_dim:
            scale = max_preview_dim / max(h, w)
            preview = cv2.resize(panorama, (int(w * scale), int(h * scale)))
            preview_path = os.path.splitext(output_path)[0] + "_preview.jpg"
            cv2.imwrite(preview_path, preview)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        
        return output_path

def create_html_viewer(panorama_path):
    """Create an HTML viewer for exploring the panorama"""
    html_path = os.path.splitext(panorama_path)[0] + "_viewer.html"
    panorama_filename = os.path.basename(panorama_path)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>CSCI 576 Panorama Viewer</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #333;
            font-family: Arial, sans-serif;
        }}
        #viewer-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}
        #panorama {{
            position: absolute;
            cursor: grab;
        }}
        #panorama:active {{
            cursor: grabbing;
        }}
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background-color: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            color: white;
            display: flex;
            align-items: center;
            z-index: 100;
        }}
        #controls button {{
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 8px 16px;
            margin: 0 5px;
            cursor: pointer;
            border-radius: 3px;
        }}
        #zoom-level {{
            margin: 0 10px;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
            max-width: 300px;
            z-index: 100;
        }}
    </style>
</head>
<body>
    <div id="viewer-container">
        <img id="panorama" src="{panorama_filename}" alt="Panorama Image">
    </div>
    <div class="info-panel">
        <h3>CSCI 576 Multimedia Project</h3>
        <p>Ultra-high-resolution panorama created from multiple video frames with gap filling</p>
        <p>Use mouse wheel to zoom and drag to pan.</p>
    </div>
    <div id="controls">
        <button id="zoom-in">Zoom In (+)</button>
        <button id="zoom-out">Zoom Out (-)</button>
        <button id="reset">Reset View</button>
        <span id="zoom-level">Zoom: 100%</span>
    </div>

    <script>
        window.onload = function() {{
            const container = document.getElementById('viewer-container');
            const panorama = document.getElementById('panorama');
            const zoomInBtn = document.getElementById('zoom-in');
            const zoomOutBtn = document.getElementById('zoom-out');
            const resetBtn = document.getElementById('reset');
            const zoomLevelDisplay = document.getElementById('zoom-level');
            
            let scale = 1;
            let offsetX = 0;
            let offsetY = 0;
            let startX = 0;
            let startY = 0;
            let isDragging = false;
            
            // Center the image initially
            panorama.onload = function() {{
                offsetX = (container.clientWidth - panorama.width * scale) / 2;
                offsetY = (container.clientHeight - panorama.height * scale) / 2;
                updateImagePosition();
            }};
            
            function updateImagePosition() {{
                panorama.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px) scale(${{scale}})`;
                panorama.style.transformOrigin = '0 0';
                zoomLevelDisplay.textContent = `Zoom: ${{Math.round(scale * 100)}}%`;
            }}
            
            // Mouse events for dragging
            panorama.addEventListener('mousedown', function(e) {{
                isDragging = true;
                startX = e.clientX - offsetX;
                startY = e.clientY - offsetY;
                e.preventDefault();
            }});
            
            document.addEventListener('mousemove', function(e) {{
                if (isDragging) {{
                    offsetX = e.clientX - startX;
                    offsetY = e.clientY - startY;
                    updateImagePosition();
                    e.preventDefault();
                }}
            }});
            
            document.addEventListener('mouseup', function() {{
                isDragging = false;
            }});
            
            // Zoom with mouse wheel
            container.addEventListener('wheel', function(e) {{
                e.preventDefault();
                const delta = e.deltaY > 0 ? -0.1 : 0.1;
                const mouseX = e.clientX;
                const mouseY = e.clientY;
                
                const imgX = (mouseX - offsetX) / scale;
                const imgY = (mouseY - offsetY) / scale;
                
                scale = Math.max(0.1, Math.min(10, scale + delta));
                
                offsetX = mouseX - imgX * scale;
                offsetY = mouseY - imgY * scale;
                
                updateImagePosition();
            }});
            
            // Button controls
            zoomInBtn.addEventListener('click', function() {{
                const centerX = container.clientWidth / 2;
                const centerY = container.clientHeight / 2;
                
                const imgX = (centerX - offsetX) / scale;
                const imgY = (centerY - offsetY) / scale;
                
                scale = Math.min(10, scale + 0.2);
                
                offsetX = centerX - imgX * scale;
                offsetY = centerY - imgY * scale;
                
                updateImagePosition();
            }});
            
            zoomOutBtn.addEventListener('click', function() {{
                const centerX = container.clientWidth / 2;
                const centerY = container.clientHeight / 2;
                
                const imgX = (centerX - offsetX) / scale;
                const imgY = (centerY - offsetY) / scale;
                
                scale = Math.max(0.1, scale - 0.2);
                
                offsetX = centerX - imgX * scale;
                offsetY = centerY - imgY * scale;
                
                updateImagePosition();
            }});
            
            resetBtn.addEventListener('click', function() {{
                scale = 1;
                offsetX = (container.clientWidth - panorama.width * scale) / 2;
                offsetY = (container.clientHeight - panorama.height * scale) / 2;
                updateImagePosition();
            }});
        }};
    </script>
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path

def main():
    parser = argparse.ArgumentParser(
        description="Create complete ultra-high-resolution images from multiple videos with gap filling")
    
    parser.add_argument("--videos", nargs="+", required=True,
                      help="List of video files to process")
    parser.add_argument("--output", default="complete_panorama.jpg",
                      help="Output filename for the panorama")
    parser.add_argument("--sample-rate", type=int, default=3,
                      help="Frame sampling rate (default: 3 - lower means more frames)")
    parser.add_argument("--debug", action="store_true",
                      help="Save intermediate results for debugging")
    parser.add_argument("--create-viewer", action="store_true",
                      help="Create HTML viewer for exploring the panorama")
    
    args = parser.parse_args()
    
    # Validate video paths
    for video_path in args.videos:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1
    
    # Create stitcher
    stitcher = CompletePanoramaStitcher(debug=args.debug)
    
    try:
        # Process videos
        result = stitcher.process_videos(args.videos, args.output, args.sample_rate)
        
        if result:
            logger.info(f"Successfully created panorama: {result}")
            
            # Create HTML viewer if requested
            if args.create_viewer:
                viewer_path = create_html_viewer(result)
                logger.info(f"Created HTML viewer: {viewer_path}")
            
            return 0
        else:
            logger.error("Failed to create panorama")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error processing videos: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
