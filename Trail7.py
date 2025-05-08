#!/usr/bin/env python3
# Ultra-High-Resolution Image Stitcher with GUI
# Based on the original working logic with added GUI interface

import cv2
import numpy as np
import os
import argparse
import time
import logging
import sys
import gc
from skimage import transform as tf
from scipy import ndimage
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class PanoramaGUI:
    """GUI for the panorama stitcher application"""
    def __init__(self, stitcher):
        self.root = tk.Tk()
        self.root.title("HD Panorama Stitcher")
        self.root.geometry("900x700")
        
        # Reference to the stitcher
        self.stitcher = stitcher
        
        # Variables
        self.video_files = []
        self.current_directory = tk.StringVar(value=os.path.expanduser("~"))
        self.output_file = tk.StringVar(value="panorama.jpg")
        self.sample_rate = tk.IntVar(value=8)  # Default from original code
        self.status_text = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)
        self.is_processing = False
        self.debug = tk.BooleanVar(value=False)
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Directory selection
        dir_frame = ttk.LabelFrame(main_frame, text="Select Video Location", padding="10")
        dir_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(dir_frame, text="Current Directory:").grid(row=0, column=0, sticky=tk.W)
        dir_entry = ttk.Entry(dir_frame, textvariable=self.current_directory, width=60)
        dir_entry.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=5)
        ttk.Button(dir_frame, text="Refresh", command=self.refresh_file_list).grid(row=0, column=3, padx=5)
        
        # File browser frame
        browser_frame = ttk.LabelFrame(main_frame, text="Available Videos", padding="10")
        browser_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create file list with scrollbar
        self.file_listbox = tk.Listbox(browser_frame, height=8, selectmode=tk.EXTENDED)
        file_scrollbar = ttk.Scrollbar(browser_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_scrollbar.set)
        
        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        file_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Add buttons for file operations
        file_buttons_frame = ttk.Frame(browser_frame)
        file_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(file_buttons_frame, text="Add Selected →", command=self.add_selected_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_buttons_frame, text="Add All Videos", command=self.add_all_videos).pack(side=tk.LEFT, padx=5)
        
        # Selected videos frame
        selected_frame = ttk.LabelFrame(main_frame, text="Selected Videos for Panorama", padding="10")
        selected_frame.grid(row=1, column=2, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.video_listbox = tk.Listbox(selected_frame, height=8, selectmode=tk.EXTENDED)
        video_scrollbar = ttk.Scrollbar(selected_frame, orient=tk.VERTICAL, command=self.video_listbox.yview)
        self.video_listbox.configure(yscrollcommand=video_scrollbar.set)
        
        self.video_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Remove buttons
        remove_buttons_frame = ttk.Frame(selected_frame)
        remove_buttons_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(remove_buttons_frame, text="← Remove Selected", command=self.remove_videos).pack(side=tk.LEFT, padx=5)
        ttk.Button(remove_buttons_frame, text="Clear All", command=self.clear_all_videos).pack(side=tk.LEFT, padx=5)
        
        # Configure column weights for browser
        browser_frame.columnconfigure(0, weight=1)
        selected_frame.columnconfigure(0, weight=1)
        
        # Output file
        ttk.Label(main_frame, text="Output File:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_file, width=50).grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=2, column=3, padx=5)
        
        # Sample rate
        ttk.Label(main_frame, text="Sample Rate:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Scale(main_frame, from_=1, to=30, variable=self.sample_rate, orient=tk.HORIZONTAL).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(main_frame, textvariable=self.sample_rate).grid(row=3, column=2, sticky=tk.W, pady=5)
        
        # Debug mode
        ttk.Checkbutton(main_frame, text="Debug Mode", variable=self.debug).grid(row=3, column=3, sticky=tk.W, pady=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        
        # Status
        ttk.Label(main_frame, textvariable=self.status_text).grid(row=5, column=0, columnspan=4, sticky=tk.W, pady=5)
        
        # Preview area
        self.preview_label = ttk.Label(main_frame, text="Preview will appear here")
        self.preview_label.grid(row=6, column=0, columnspan=4, pady=10)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing)
        self.process_button.grid(row=7, column=0, columnspan=4, pady=10)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Refresh file list on startup
        self.refresh_file_list()
    
    def browse_directory(self):
        """Browse for a directory containing video files"""
        directory = filedialog.askdirectory(
            title="Select Directory Containing Videos",
            initialdir=self.current_directory.get()
        )
        if directory:
            self.current_directory.set(directory)
            self.refresh_file_list()
    
    def refresh_file_list(self):
        """Refresh the list of available video files in the current directory"""
        self.file_listbox.delete(0, tk.END)
        
        try:
            directory = self.current_directory.get()
            if os.path.exists(directory):
                # Get all video files in the directory
                video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
                files = []
                
                for file in os.listdir(directory):
                    if file.lower().endswith(video_extensions):
                        files.append(file)
                
                # Sort files
                files.sort()
                
                # Add to listbox
                for file in files:
                    self.file_listbox.insert(tk.END, file)
                
                if not files:
                    self.file_listbox.insert(tk.END, "No video files found in this directory")
            else:
                self.file_listbox.insert(tk.END, "Directory does not exist")
        
        except Exception as e:
            self.file_listbox.insert(tk.END, f"Error: {str(e)}")
    
    def add_selected_videos(self):
        """Add selected videos from the file list to the processing list"""
        selected_indices = self.file_listbox.curselection()
        directory = self.current_directory.get()
        
        for index in selected_indices:
            filename = self.file_listbox.get(index)
            if filename and not filename.startswith("No video files") and not filename.startswith("Error:"):
                full_path = os.path.join(directory, filename)
                if full_path not in self.video_files:
                    self.video_files.append(full_path)
                    self.video_listbox.insert(tk.END, filename)
    
    def add_all_videos(self):
        """Add all videos from the current directory to the processing list"""
        directory = self.current_directory.get()
        
        for index in range(self.file_listbox.size()):
            filename = self.file_listbox.get(index)
            if filename and not filename.startswith("No video files") and not filename.startswith("Error:"):
                full_path = os.path.join(directory, filename)
                if full_path not in self.video_files:
                    self.video_files.append(full_path)
                    self.video_listbox.insert(tk.END, filename)
    
    def clear_all_videos(self):
        """Clear all videos from the selected list"""
        self.video_files.clear()
        self.video_listbox.delete(0, tk.END)
    
    def remove_videos(self):
        """Remove selected videos from the list"""
        selected_indices = self.video_listbox.curselection()
        for index in reversed(selected_indices):
            self.video_listbox.delete(index)
            del self.video_files[index]
    
    def browse_output(self):
        """Browse for output file location"""
        file_path = filedialog.asksaveasfilename(
            title="Save Panorama As",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All Files", "*.*")]
        )
        if file_path:
            self.output_file.set(file_path)
    
    def update_preview(self, image):
        """Update the preview image"""
        try:
            # Resize image for preview
            height, width = image.shape[:2]
            max_size = 400
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            
            resized = cv2.resize(image, (new_width, new_height))
            # Convert from BGR to RGB
            image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Convert to PhotoImage
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            
            # Update label
            self.preview_label.configure(image=image_tk)
            self.preview_label.image = image_tk
        except Exception as e:
            logger.error(f"Error updating preview: {e}")
    
    def start_processing(self):
        """Start the panorama processing"""
        if not self.video_files:
            messagebox.showerror("Error", "Please add at least one video file")
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Processing is already in progress")
            return
        
        self.is_processing = True
        self.process_button.configure(state='disabled')
        
        # Update stitcher debug mode
        self.stitcher.debug = self.debug.get()
        
        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_videos)
        thread.daemon = True
        thread.start()
    
    def process_videos(self):
        """Process videos in a separate thread"""
        try:
            result = self.stitcher.process_videos(
                self.video_files,
                self.output_file.get(),
                self.sample_rate.get()
            )
            
            if result:
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Panorama saved to {result}"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to create panorama"))
        
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error processing videos: {str(e)}"))
        
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_button.configure(state='normal'))
    
    def update_status(self, text, progress):
        """Update GUI from processing thread"""
        self.root.after(0, lambda: self.status_text.set(text))
        self.root.after(0, lambda: self.progress_var.set(progress))
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

# Your original CompletePanoramaStitcher class with GUI callbacks added
class CompletePanoramaStitcher:
    def __init__(self, output_dir="output", temp_dir="temp_frames", 
                 max_resolution=2000, debug=False, gui=None):
        """Initialize the complete panorama stitcher with gap filling capabilities"""
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.max_resolution = max_resolution
        self.debug = debug
        self.gui = gui  # Reference to GUI for callbacks
        
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
            self.detector = cv2.cuda.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=10)
            self.use_gpu = True
            logger.info("Using GPU-accelerated SIFT")
        except (cv2.error, AttributeError):
            # Fallback to CPU version with optimized parameters
            self.detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=10)
            self.use_gpu = False
            logger.info("Using CPU-based SIFT")

    def update_status(self, text, progress):
        """Update status through GUI if available"""
        if self.gui:
            self.gui.update_status(text, progress)
        logger.info(f"{text} - {progress}%")

    def extract_frames(self, video_path, sample_rate=8):
        """Extract frames from a video file in parallel at the specified sample rate"""
        self.update_status(f"Extracting frames from {os.path.basename(video_path)}", 0)
        
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
        
        # Extract frames sequentially for stability (parallel can cause issues with video)
        frames = []
        for i, frame_idx in enumerate(frames_to_extract):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_path = os.path.join(self.temp_dir, f"{video_name}_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames.append(frame_path)
            
            if i % 10 == 0:
                self.update_status(f"Extracting frames from {video_name}", 
                                 int((i + 1) / len(frames_to_extract) * 100))
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    # All other methods remain exactly the same as in your original code
    def find_matching_features(self, img1, img2):
        """Find matching features between two images with GPU acceleration if available"""
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE to improve feature detection in difficult areas
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)
        
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
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    if m.queryIdx < len(kp1) and m.trainIdx < len(kp2):
                        good_matches.append(m)
        
        if len(good_matches) < 10:
            return None, None, 0
        
        # Extract coordinates of matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC with more iterations for better accuracy
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0, maxIters=2000)
        
        if H is None:
            return None, None, 0
            
        matched_count = np.sum(mask) if mask is not None else 0
        
        return H, mask, matched_count

    def build_image_graph(self, image_paths):
        """Build a graph of images based on feature matches between pairs"""
        self.update_status("Building image relationship graph", 0)
        
        n_images = len(image_paths)
        
        # Load all images to memory
        images = []
        for i, path in enumerate(image_paths):
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
            
            if i % 10 == 0:
                self.update_status("Loading images", int((i + 1) / n_images * 100))

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
        for idx, (i, j) in enumerate(pairs_to_check):
            # Find homography between images
            H, mask, matched_count = self.find_matching_features(images[i], images[j])
            
            if H is not None and matched_count >= 10:
                # Store edge in graph
                graph[i].append((j, matched_count))
                graph[j].append((i, matched_count))
                
                # Store homography
                homographies[(i, j)] = H
                homographies[(j, i)] = np.linalg.inv(H)
            
            if idx % 10 == 0:
                self.update_status("Finding matches", int((idx + 1) / len(pairs_to_check) * 100))
        
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

    # Continue with all other methods from your original code unchanged...
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
        self.update_status("Finding optimal stitching order", 0)
        
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
        self.update_status("Computing global transformations", 0)
        
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
        self.update_status("Estimating panorama dimensions", 0)
        
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
        self.update_status("Creating initial panorama", 0)
        
        # Create empty panorama and accumulation buffers
        panorama = np.zeros((height, width, 3), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create weight maps for better blending
        weight_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        # Sort images by distance from center for better blending order
        center_idx = len(images) // 2
        image_indices = [(i, img) for i, img in enumerate(images) if img is not None and i in global_transforms]
        image_indices.sort(key=lambda x: abs(x[0] - center_idx))
        
        # Warp and blend each image using multi-band blending
        for idx_in_list, (idx, img) in enumerate(image_indices):
            # Apply sharpening filter to images to enhance details
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_sharp = cv2.filter2D(img, -1, kernel)
            img_enhanced = cv2.addWeighted(img, 0.7, img_sharp, 0.3, 0)
            
            # Warp image
            warped = cv2.warpPerspective(img_enhanced.astype(np.float32), global_transforms[idx], (width, height))
            
            # Create mask for this image
            img_mask = np.any(warped > 0, axis=2).astype(np.uint8)
            
            # Create advanced distance map for better blending
            dist_mask = cv2.distanceTransform(img_mask, cv2.DIST_L2, 5)
            dist_mask = cv2.GaussianBlur(dist_mask, (21, 21), 0)
            
            # Normalize distance mask
            max_dist = np.max(dist_mask) if np.max(dist_mask) > 0 else 1
            dist_mask = dist_mask / max_dist
            
            # Apply pyramid blending for smoother transitions
            if np.any(mask > 0):
                # Create Gaussian pyramid for multi-band blending
                levels = 4
                gp_warped = [warped]
                gp_panorama = [panorama]
                
                # Convert single channel mask to 3 channels for proper multiplication
                dist_mask_3ch = np.repeat(dist_mask[:, :, np.newaxis], 3, axis=2)
                gp_mask = [dist_mask_3ch]
                
                for i in range(levels):
                    gp_warped.append(cv2.pyrDown(gp_warped[i]))
                    gp_panorama.append(cv2.pyrDown(gp_panorama[i]))
                    gp_mask.append(cv2.pyrDown(gp_mask[i]))
                
                # Create Laplacian pyramids
                lp_warped = [gp_warped[levels]]
                lp_panorama = [gp_panorama[levels]]
                
                for i in range(levels, 0, -1):
                    size = (gp_warped[i-1].shape[1], gp_warped[i-1].shape[0])
                    lp_warped.append(gp_warped[i-1] - cv2.pyrUp(gp_warped[i], dstsize=size))
                    lp_panorama.append(gp_panorama[i-1] - cv2.pyrUp(gp_panorama[i], dstsize=size))
                
                # Blend pyramids
                lp_blended = []
                for lw, lp, gm in zip(lp_warped, lp_panorama, gp_mask[::-1]):
                    blended = lw * gm + lp * (1 - gm)
                    lp_blended.append(blended)
                
                # Reconstruct blended image
                blended_img = lp_blended[0]
                for i in range(1, levels + 1):
                    size = (lp_blended[i].shape[1], lp_blended[i].shape[0])
                    blended_img = cv2.pyrUp(blended_img, dstsize=size) + lp_blended[i]
                
                # Update panorama
                panorama = blended_img
            else:
                # First image, just copy
                panorama = warped
            
            # Update masks
            mask = np.maximum(mask, img_mask)
            weight_map = np.maximum(weight_map, dist_mask)
            
            # Update preview in GUI
            if self.gui and self.gui.preview_label and idx_in_list % 5 == 0:
                preview = np.clip(panorama, 0, 255).astype(np.uint8)
                self.gui.root.after(0, lambda p=preview.copy(): self.gui.update_preview(p))
            
            self.update_status("Blending images", int((idx_in_list + 1) / len(image_indices) * 100))
        
        # Convert back to uint8
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        
        # Store the true panorama mask without any filled gaps for later use
        self.true_content_mask = mask.copy()
        
        return panorama, mask

    def find_seams_and_inconsistencies(self, panorama, mask):
        """Find seam lines and inconsistencies in the panorama"""
        self.update_status("Detecting seams and inconsistencies", 0)
        
        # Create a simplified approach that avoids the error
        # Just return the edge areas of the mask as potential seams
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        seam_mask = dilated - eroded
        
        return seam_mask

    def fill_panorama_gaps_simple(self, panorama, mask):
        """Fill gaps in the panorama using simple inpainting"""
        self.update_status("Filling gaps with simple inpainting", 0)
        
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

    def apply_clean_background(self, panorama):
        """Apply a clean white background outside the actual content area"""
        self.update_status("Applying clean white background", 0)
        
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
        
        # Apply clean white background outside the actual content area
        final_panorama = self.apply_clean_background(filled_panorama)
        
        return final_panorama

    def clean_temp_files(self):
        """Remove temporary files to free disk space"""
        self.update_status("Cleaning temporary files", 0)
        
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
            for i, video_path in enumerate(video_paths):
                frames = self.extract_frames(video_path, sample_rate)
                all_frames.extend(frames)
                self.update_status(f"Extracted frames from video {i+1}/{len(video_paths)}", 
                                 int((i + 1) / len(video_paths) * 100))
            
            if not all_frames:
                logger.error("No frames were extracted from videos")
                return None
            
            # Create panorama
            panorama = self.create_complete_panorama(all_frames)
            
            if panorama is None:
                logger.error("Failed to create panorama")
                return None
            
            # Save the final panorama
            self.update_status("Saving final panorama", 95)
            cv2.imwrite(output_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Create a preview version
            h, w = panorama.shape[:2]
            max_preview_dim = 1500
            if max(h, w) > max_preview_dim:
                scale = max_preview_dim / max(h, w)
                preview = cv2.resize(panorama, (int(w * scale), int(h * scale)))
                preview_path = os.path.splitext(output_path)[0] + "_preview.jpg"
                cv2.imwrite(preview_path, preview)
            
            # Clean up temporary files
            if not self.debug:
                self.clean_temp_files()
            
            elapsed_time = time.time() - start_time
            self.update_status(f"Total processing time: {elapsed_time:.2f} seconds", 100)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error in panorama creation: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the application"""
    # Create stitcher
    stitcher = CompletePanoramaStitcher()
    
    # Create and run GUI
    gui = PanoramaGUI(stitcher)
    stitcher.gui = gui  # Set reference to GUI
    gui.run()

if __name__ == "__main__":
    main()
