#!/usr/bin/env python3
# Complete Ultra-High-Resolution Image Stitcher with Gap Filling
# CSCI 576 Multimedia Project - Spring 2025
# Speed-optimized version - runs in 10-15 mins

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
import multiprocessing
from joblib import Parallel, delayed

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
        self.initialize_feature_detector()
        
        # Initialize feature matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Store homographies for potential reuse
        self.homographies = {}
        
        # Store the original content mask (before gap filling)
        self.true_content_mask = None

    def initialize_feature_detector(self):
        """Initialize the feature detector with GPU acceleration if available"""
        try:
            # Try to use GPU acceleration for SIFT
            self.detector = cv2.cuda.SIFT_create(nfeatures=3000)
            self.use_gpu = True
            logger.info("Using GPU-accelerated SIFT")
        except (cv2.error, AttributeError):
            # Fallback to CPU version
            self.detector = cv2.SIFT_create(nfeatures=3000)
            self.use_gpu = False
            logger.info("Using CPU-based SIFT")

    def extract_frames(self, video_path, sample_rate=8):
        """Extract frames from a video file in parallel at the specified sample rate"""
        logger.info(f"Extracting frames from {video_path} with sample rate {sample_rate}")
        
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
        
        # Calculate frames to extract
        frames_to_extract = list(range(0, total_frames, sample_rate))
        expected_frames = len(frames_to_extract)
        logger.info(f"Will extract approximately {expected_frames} frames")
        
        # Define function to extract a specific frame
        def extract_single_frame(frame_idx, video_path, output_dir, video_name):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                frame_path = os.path.join(output_dir, f"{video_name}_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return frame_path
            return None
        
        # Use parallel processing
        num_cores = multiprocessing.cpu_count()
        logger.info(f"Using {num_cores} cores for parallel frame extraction")
        
        frames = Parallel(n_jobs=num_cores)(
            delayed(extract_single_frame)(
                frame_idx, video_path, self.temp_dir, video_name
            ) for frame_idx in tqdm(frames_to_extract, desc=f"Extracting frames from {video_name}")
        )
        
        # Filter out None values (failed extractions)
        frames = [f for f in frames if f is not None]
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    def find_matching_features(self, img1, img2):
        """Find matching features between two images with GPU acceleration if available"""
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        if hasattr(self, 'use_gpu') and self.use_gpu:
            try:
                # Upload images to GPU
                gpu_gray1 = cv2.cuda_GpuMat()
                gpu_gray2 = cv2.cuda_GpuMat()
                gpu_gray1.upload(gray1)
                gpu_gray2.upload(gray2)
                
                # Detect keypoints and compute descriptors on GPU
                kp1, des1 = self.detector.detectAndCompute(gpu_gray1)
                kp2, des2 = self.detector.detectAndCompute(gpu_gray2)
                
                # Download descriptors from GPU
                des1 = des1.download()
                des2 = des2.download()
            except Exception as e:
                # Fall back to CPU if GPU fails
                logger.warning(f"GPU processing failed: {e}. Falling back to CPU.")
                kp1, des1 = self.detector.detectAndCompute(gray1, None)
                kp2, des2 = self.detector.detectAndCompute(gray2, None)
        else:
            # CPU version
            kp1, des1 = self.detector.detectAndCompute(gray1, None)
            kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        # Continue with matching as before
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
        
        # Load all images to memory
        images = []
        for path in tqdm(image_paths, desc="Loading images"):
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

        # Create empty graph
        graph = {i: [] for i in range(n_images)}
        homographies = {}
        
        # HYBRID APPROACH: Check sequential frames plus keyframes
        logger.info("Using hybrid matching approach")
        pairs_to_check = []
        window_size = 10  # Only match with nearby frames
        
        # 1. Add sequential neighboring pairs (temporal matching)
        for i in range(n_images):
            if images[i] is None:
                continue
            # Check nearby frames within window_size
            for j in range(i+1, min(i+window_size, n_images)):
                if images[j] is None:
                    continue
                pairs_to_check.append((i, j))
        
        # 2. Add keyframe pairs (spatial matching) - sample at regular intervals
        keyframe_interval = 20  # Use every 20th frame as a keyframe
        keyframes = [i for i in range(0, n_images, keyframe_interval) if i < n_images and images[i] is not None]
        
        # Match each keyframe with all other keyframes
        for i, kf1 in enumerate(keyframes):
            for kf2 in keyframes[i+1:]:
                pairs_to_check.append((kf1, kf2))
        
        # Remove duplicates
        pairs_to_check = list(set(pairs_to_check))
        
        logger.info(f"Checking {len(pairs_to_check)} pairs")
        
        # Process pairs sequentially to avoid pickling issues
        for i, j in tqdm(pairs_to_check, desc="Finding matches"):
            # Find homography between images
            H, mask, matched_count = self.find_matching_features(images[i], images[j])
            
            if H is not None and matched_count >= 10:
                # Store edge in graph
                graph[i].append((j, matched_count))
                graph[j].append((i, matched_count))
                
                # Store homography
                homographies[(i, j)] = H
                homographies[(j, i)] = np.linalg.inv(H)
        
        # Verify graph connectivity - if there are multiple components, try to connect them
        if not self.is_graph_connected(graph, n_images):
            logger.warning("Graph is not fully connected - adding additional matches to connect components")
            components = self.find_connected_components(graph, n_images)
            
            # Try to connect different components
            if len(components) > 1:
                for i, comp1 in enumerate(components):
                    for comp2 in components[i+1:]:
                        # Find best pair between components
                        best_pair = None
                        max_matches = 0
                        
                        # Sample nodes from each component to check
                        nodes1 = list(comp1)[:min(3, len(comp1))]  # Check up to 3 nodes
                        nodes2 = list(comp2)[:min(3, len(comp2))]
                        
                        for node1 in nodes1:
                            for node2 in nodes2:
                                if images[node1] is None or images[node2] is None:
                                    continue
                                    
                                # Find homography between images
                                H, mask, matched_count = self.find_matching_features(images[node1], images[node2])
                                
                                if H is not None and matched_count > max_matches:
                                    max_matches = matched_count
                                    best_pair = (node1, node2, H)
                        
                        # If found a connection, add it to the graph
                        if best_pair:
                            node1, node2, H = best_pair
                            graph[node1].append((node2, max_matches))
                            graph[node2].append((node1, max_matches))
                            
                            homographies[(node1, node2)] = H
                            homographies[(node2, node1)] = np.linalg.inv(H)
                            
                            logger.info(f"Added connection between components: {node1} <-> {node2} with {max_matches} matches")
        
        return graph, homographies, images

    def is_graph_connected(self, graph, n_images):
        """Check if the graph is connected"""
        if not graph:
            return False
        
        # Start BFS from first non-empty node
        start_node = None
        for i in range(n_images):
            if i in graph and graph[i]:
                start_node = i
                break
        
        if start_node is None:
            return False
        
        # BFS
        visited = {start_node}
        queue = [start_node]
        
        while queue:
            node = queue.pop(0)
            for neighbor, _ in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Check if all nodes with edges are visited
        for i in range(n_images):
            if i in graph and graph[i] and i not in visited:
                return False
                
        return True

    def find_connected_components(self, graph, n_images):
        """Find all connected components in the graph"""
        visited = set()
        components = []
        
        for i in range(n_images):
            if i in graph and graph[i] and i not in visited:
                # Start BFS from this node
                component = set()
                queue = [i]
                component.add(i)
                visited.add(i)
                
                while queue:
                    node = queue.pop(0)
                    for neighbor, _ in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        return components

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

    def estimate_transform_from_neighbors(self, missing_idx, global_transforms, graph):
        """Estimate transformation for a node with missing direct connection to the tree"""
        logger.info(f"Estimating transform for node {missing_idx} from neighbors")
        
        # Get all neighbors that have valid transforms
        neighbors = []
        for neighbor, weight in graph[missing_idx]:
            if neighbor in global_transforms and global_transforms[neighbor] is not None:
                neighbors.append((neighbor, weight))
        
        if not neighbors:
            logger.warning(f"No valid neighbors found for node {missing_idx}")
            return None
        
        # Sort neighbors by matching weight (descending)
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Try to estimate transform from each neighbor
        for neighbor, weight in neighbors:
            if (missing_idx, neighbor) in self.homographies:
                H_relative = self.homographies[(missing_idx, neighbor)]
                estimated_transform = global_transforms[neighbor] @ np.linalg.inv(H_relative)
                logger.info(f"Estimated transform for node {missing_idx} from neighbor {neighbor}")
                return estimated_transform
        
        return None

    def compute_global_transforms(self, spanning_tree, homographies, n_images, graph):
        """Compute global transformation matrices for all images based on the spanning tree"""
        logger.info("Computing global transformations...")
        
        # Save homographies for other methods
        self.homographies = homographies
        
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
        
        # Try to estimate transforms for nodes not in the spanning tree
        for i in range(n_images):
            if i not in visited and i in graph:
                estimated_transform = self.estimate_transform_from_neighbors(i, global_transforms, graph)
                if estimated_transform is not None:
                    global_transforms[i] = estimated_transform
                    logger.info(f"Added node {i} with estimated transform")
        
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
        
        # Create a weight map for blending
        weight_map = np.zeros((height, width), dtype=np.float32)
        
        # Sort images by area (process larger areas first as they likely form the core of the panorama)
        image_indices = [(i, img) for i, img in enumerate(images) if img is not None and i in global_transforms]
        image_indices.sort(key=lambda x: x[1].shape[0] * x[1].shape[1], reverse=True)
        
        # Warp and blend each image
        for idx, img in image_indices:
            # Warp image
            warped = cv2.warpPerspective(img, global_transforms[idx], (width, height))
            
            # Create mask for this image
            img_mask = np.any(warped > 0, axis=2).astype(np.uint8)
            
            # Create distance map from the edge (for better blending)
            dist_mask = cv2.distanceTransform(img_mask, cv2.DIST_L2, 3)
            # Normalize to [0,1]
            max_dist = np.max(dist_mask) if np.max(dist_mask) > 0 else 1
            dist_mask = dist_mask / max_dist
            
            # Update the panorama with feathered blending
            existing_mask = mask > 0
            
            # Where there is no existing image, just copy the new image
            panorama[~existing_mask & (img_mask > 0)] = warped[~existing_mask & (img_mask > 0)]
            
            # Where the images overlap, blend based on the distance maps
            overlap_region = existing_mask & (img_mask > 0)
            if np.any(overlap_region):
                # Weight is higher for pixels farther from edges
                weight_new = dist_mask[overlap_region]
                weight_existing = weight_map[overlap_region]
                
                # Normalize the weights to sum to 1
                total_weight = weight_new + weight_existing
                total_weight[total_weight == 0] = 1  # Avoid division by zero
                
                weight_new = weight_new / total_weight
                weight_existing = weight_existing / total_weight
                
                # Blend the images
                blended = np.zeros_like(warped)
                blended[overlap_region] = (weight_new[:, np.newaxis] * warped[overlap_region] + 
                                         weight_existing[:, np.newaxis] * panorama[overlap_region])
                
                panorama[overlap_region] = blended[overlap_region]
            
            # Update the mask and weight map
            mask = np.maximum(mask, img_mask)
            weight_map = np.maximum(weight_map, dist_mask)
        
        # Store the true panorama mask without any filled gaps for later use
        self.true_content_mask = mask.copy()
        
        return panorama, mask

    def find_seams_and_inconsistencies(self, panorama, mask):
        """Find seam lines and inconsistencies in the panorama"""
        logger.info("Detecting seams and inconsistencies...")
        
        # Create a simplified approach that avoids the error
        # Just return the edge areas of the mask as potential seams
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        seam_mask = dilated - eroded
        
        return seam_mask

    def fill_panorama_gaps_simple(self, panorama, mask):
        """Fill gaps in the panorama using simple inpainting"""
        logger.info("Filling gaps with simple inpainting...")
        
        # Create a binary mask for holes (inverted mask)
        holes = (mask == 0).astype(np.uint8)
        
        # If no holes, return the original panorama
        if not np.any(holes):
            logger.info("No gaps found in the panorama")
            return panorama
        
        # Dilate the holes slightly to create better transitions
        holes_dilated = cv2.dilate(holes, np.ones((3,3), np.uint8), iterations=1)
        
        # Use simple inpainting for holes
        try:
            filled_panorama = cv2.inpaint(panorama, holes_dilated, 3, cv2.INPAINT_TELEA)
            return filled_panorama
        except Exception as e:
            logger.error(f"Error during inpainting: {e}")
            # If inpainting fails, try a different method
            try:
                # Use a simpler method - just blur the boundaries
                kernel_size = 9
                blurred = cv2.GaussianBlur(panorama, (kernel_size, kernel_size), 0)
                filled_panorama = np.copy(panorama)
                filled_panorama[holes_dilated > 0] = blurred[holes_dilated > 0]
                return filled_panorama
            except:
                # If all else fails, return the original
                return panorama

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

    def apply_clean_background(self, panorama):
        """Apply a clean white background outside the actual content area"""
        logger.info("Applying clean white background to non-content areas")
        
        if not hasattr(self, 'true_content_mask') or self.true_content_mask is None:
            logger.warning("No content mask available, cannot apply clean background")
            return panorama
        
        # Create white background image
        white_bg = np.full_like(panorama, 255)  # White background (all 255s)
        
        # Expand mask to 3 channels for color image
        mask_3ch = np.repeat(self.true_content_mask[:, :, np.newaxis], 3, axis=2)
        
        # Composite panorama on white background
        result = np.where(mask_3ch > 0, panorama, white_bg)
        
        return result

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
        global_transforms = self.compute_global_transforms(spanning_tree, homographies, len(images), graph)
        
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
        
        # Fill gaps in the panorama using simple method to avoid errors
        filled_panorama = self.fill_panorama_gaps_simple(panorama, mask)
        
        # Enhance the panorama quality
        final_panorama = self.enhance_panorama_quality(filled_panorama)
        
        # Apply clean white background outside the actual content area
        final_panorama = self.apply_clean_background(final_panorama)
        
        return final_panorama

    def clean_temp_files(self):
        """Remove temporary files to free disk space"""
        logger.info("Cleaning temporary files...")
        
        if os.path.exists(self.temp_dir):
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")

    def process_videos(self, video_paths, output_path="panorama.jpg", sample_rate=8):
        """Process multiple videos to create a complete panorama"""
        start_time = time.time()
        
        try:
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
            #h, w = panorama.shape[:2]
            #max_preview_dim = 1500
            #if max(h, w) > max_preview_dim:
                #scale = max_preview_dim / max(h, w)
                #preview = cv2.resize(panorama, (int(w * scale), int(h * scale)))
                #preview_path = os.path.splitext(output_path)[0] + "_preview.jpg"
                #cv2.imwrite(preview_path, preview)
            
            # Clean up temporary files
            if not self.debug:
                self.clean_temp_files()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error in panorama creation: {e}")
            import traceback
            traceback.print_exc()
            return None



def main():
    parser = argparse.ArgumentParser(
        description="Create complete ultra-high-resolution images from multiple videos with gap filling")
    
    parser.add_argument("--videos", nargs="+", required=True,
                      help="List of video files to process")
    parser.add_argument("--output", default="complete_panorama.jpg",
                      help="Output filename for the panorama")
    parser.add_argument("--sample-rate", type=int, default=8,
                      help="Frame sampling rate (default: 8 - higher means fewer frames)")
    parser.add_argument("--debug", action="store_true",
                      help="Save intermediate results for debugging")
    
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
