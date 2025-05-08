#!/usr/bin/env python3
# Efficient Multi-Video Stitching for Ultra-High-Resolution Images
# CSCI 576 Multimedia Project - Spring 2025

import cv2
import numpy as np
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import glob
import logging
import sys
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class MultiVideoStitcher:
    def __init__(self, output_dir="output", temp_dir="temp_frames", 
                 max_resolution=1200, debug=False):
        """Initialize the video stitcher with configuration parameters"""
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.max_resolution = max_resolution
        self.debug = debug
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Initialize feature detector
        # SIFT is good for finding feature points invariant to scale and rotation
        self.detector = cv2.SIFT_create()
        
        # Initialize feature matcher
        # FLANN based matcher is faster for large datasets
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def _resize_if_needed(self, image):
        """Resize image if it exceeds maximum resolution to conserve memory"""
        h, w = image.shape[:2]
        if max(h, w) > self.max_resolution:
            scale = self.max_resolution / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(image, new_size), scale
        return image, 1.0

    def extract_frames(self, video_path, sample_rate=15):
        """Extract frames from a video file at the specified sample rate"""
        logger.info(f"Extracting frames from {video_path} (sample rate: {sample_rate})")
        
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
        
        # Extract frames
        frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified interval
            if frame_count % sample_rate == 0:
                # Save frame to disk
                frame_path = os.path.join(self.temp_dir, f"{video_name}_{saved_count:04d}.jpg")
                
                # Optionally resize frame before saving to conserve disk space
                resized_frame, _ = self._resize_if_needed(frame)
                
                cv2.imwrite(frame_path, resized_frame)
                frames.append(frame_path)
                saved_count += 1
                
                # Print progress periodically
                if saved_count % 10 == 0:
                    logger.info(f"Extracted {saved_count} frames...")
            
            frame_count += 1
            
            # Force garbage collection periodically
            if frame_count % 100 == 0:
                gc.collect()
        
        cap.release()
        logger.info(f"Extracted {saved_count} frames from {video_path}")
        
        return frames

    def extract_frames_parallel(self, video_paths, sample_rate=15):
        """Extract frames from multiple videos in parallel"""
        logger.info(f"Extracting frames from {len(video_paths)} videos in parallel")
        
        all_frames = []
        
        # Use ThreadPoolExecutor for I/O bound tasks like video extraction
        with ThreadPoolExecutor() as executor:
            # Submit all video processing tasks
            future_to_video = {
                executor.submit(self.extract_frames, video_path, sample_rate): video_path
                for video_path in video_paths
            }
            
            # Process results as they complete
            for future in future_to_video:
                video_path = future_to_video[future]
                try:
                    frames = future.result()
                    all_frames.extend(frames)
                    logger.info(f"Finished extracting frames from {video_path}")
                except Exception as e:
                    logger.error(f"Error extracting frames from {video_path}: {e}")
        
        return all_frames

    def batch_stitch(self, frame_paths, batch_size=10):
        """Stitch images in batches to manage memory usage"""
        if len(frame_paths) <= batch_size:
            return self.stitch_images(frame_paths)
        
        logger.info(f"Stitching {len(frame_paths)} frames in batches of {batch_size}")
        
        # Process initial batch
        current_batch = frame_paths[:batch_size]
        panorama = self.stitch_images(current_batch)
        
        if panorama is None:
            logger.error("Failed to stitch initial batch")
            return None
            
        # Save intermediate panorama
        if self.debug:
            cv2.imwrite(os.path.join(self.output_dir, "batch_0.jpg"), panorama)
        
        # Process remaining images in batches
        remaining = frame_paths[batch_size:]
        total_batches = (len(remaining) + batch_size - 1) // batch_size
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(remaining))
            batch = remaining[start_idx:end_idx]
            
            logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch)} frames)")
            
            # Create temporary panorama file to save memory
            temp_panorama_path = os.path.join(self.temp_dir, "temp_panorama.jpg")
            cv2.imwrite(temp_panorama_path, panorama)
            
            # Free memory
            del panorama
            gc.collect()
            
            # Load temp panorama and stitch with new batch
            panorama = cv2.imread(temp_panorama_path)
            batch_result = self.stitch_images(batch)
            
            if batch_result is not None:
                # Stitch current panorama with batch result
                stitched_images = [panorama, batch_result]
                panorama = self.stitch_images(stitched_images, is_images=True)
                
                if panorama is None:
                    logger.error(f"Failed to stitch batch {i+1}. Using previous result.")
                    panorama = cv2.imread(temp_panorama_path)
                elif self.debug:
                    cv2.imwrite(os.path.join(self.output_dir, f"batch_{i+1}.jpg"), panorama)
            else:
                logger.warning(f"Batch {i+1} stitching failed. Continuing with previous panorama.")
                panorama = cv2.imread(temp_panorama_path)
            
            # Force garbage collection
            gc.collect()
        
        return panorama

    def stitch_images(self, image_paths, is_images=False):
        """Stitch multiple images together"""
        # Handle empty input
        if not image_paths:
            logger.error("No images provided for stitching")
            return None
            
        # Handle single image case
        if len(image_paths) == 1:
            if is_images:
                return image_paths[0]  # Already an image object
            else:
                return cv2.imread(image_paths[0])  # Load from path
                
        logger.info(f"Stitching {len(image_paths)} images")
        
        # Load images if paths are provided
        images = image_paths if is_images else []
        if not is_images:
            for path in image_paths:
                img = cv2.imread(path)
                if img is not None:
                    # Resize if needed for memory efficiency
                    img, _ = self._resize_if_needed(img)
                    images.append(img)
                else:
                    logger.warning(f"Could not read image: {path}")
        
        if len(images) < 2:
            logger.error("Not enough valid images to stitch")
            return None
            
        # Try using OpenCV's built-in stitcher first
        try:
            logger.info("Attempting stitching with OpenCV's stitcher")
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, result = stitcher.stitch(images)
            
            if status == cv2.Stitcher_OK:
                logger.info("OpenCV stitcher succeeded")
                return result
            else:
                logger.warning(f"OpenCV stitcher failed with status {status}")
        except Exception as e:
            logger.error(f"Error with OpenCV stitcher: {e}")
        
        # Fall back to custom stitching approach if OpenCV's stitcher fails
        logger.info("Falling back to custom stitching")
        try:
            return self._custom_stitch(images)
        except Exception as e:
            logger.error(f"Custom stitching failed: {e}")
            return None

    def _custom_stitch(self, images):
        """Custom stitching implementation using feature matching and homography"""
        # Start with the first image as the base panorama
        panorama = images[0]
        h_base, w_base = panorama.shape[:2]
        
        # Process each subsequent image
        for idx, img in enumerate(images[1:], 1):
            logger.info(f"Processing image {idx}/{len(images)-1} in custom stitcher")
            
            # Convert to grayscale for feature detection
            gray1 = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and compute descriptors
            kp1, des1 = self.detector.detectAndCompute(gray1, None)
            kp2, des2 = self.detector.detectAndCompute(gray2, None)
            
            # Verify we have enough features
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                logger.warning(f"Insufficient features in image {idx}")
                continue
                
            # Match descriptors
            try:
                matches = self.matcher.knnMatch(des1, des2, k=2)
            except Exception as e:
                logger.error(f"Feature matching failed: {e}")
                continue
                
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) != 2:
                    continue
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    if m.queryIdx < len(kp1) and m.trainIdx < len(kp2):  # Safety check
                        good_matches.append(m)
            
            # Need minimum number of good matches
            if len(good_matches) < 10:
                logger.warning(f"Not enough good matches for image {idx}: {len(good_matches)}")
                continue
                
            # Extract matched keypoints
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Compute homography
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                if H is None:
                    logger.warning(f"Could not find homography for image {idx}")
                    continue
                    
                # Get dimensions
                h_img, w_img = img.shape[:2]
                
                # Calculate dimensions of warped image
                corners = np.array([
                    [0, 0],
                    [0, h_img-1],
                    [w_img-1, h_img-1],
                    [w_img-1, 0]
                ], dtype=np.float32).reshape(-1, 1, 2)
                
                warped_corners = cv2.perspectiveTransform(corners, H)
                
                [x_min, y_min] = np.int32(warped_corners.min(axis=0).ravel())
                [x_max, y_max] = np.int32(warped_corners.max(axis=0).ravel())
                
                # Translation adjustment
                translation = np.array([
                    [1, 0, max(0, -x_min)],
                    [0, 1, max(0, -y_min)],
                    [0, 0, 1]
                ])
                
                H_adjusted = translation @ H
                
                # New dimensions for the panorama
                output_width = max(x_max, w_base) - min(x_min, 0)
                output_height = max(y_max, h_base) - min(y_min, 0)
                
                # Cap dimensions to avoid memory issues
                if max(output_width, output_height) > 12000:
                    logger.warning("Output size too large, capping dimensions")
                    scale = 12000 / max(output_width, output_height)
                    output_width = int(output_width * scale)
                    output_height = int(output_height * scale)
                    
                    # Scale the homography matrix
                    scale_mat = np.array([
                        [scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, 1]
                    ])
                    H_adjusted = scale_mat @ H_adjusted
                
                # Warp the new image
                warped_img = cv2.warpPerspective(
                    img, H_adjusted, (output_width, output_height))
                
                # Create a new panorama with the new dimensions
                new_panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                
                # Place the current panorama in the new one
                y_offset = max(0, -y_min)
                x_offset = max(0, -x_min)
                new_panorama[y_offset:y_offset+h_base, x_offset:x_offset+w_base] = panorama
                
                # Create a mask for blending
                mask = np.zeros((output_height, output_width), dtype=np.uint8)
                mask[y_offset:y_offset+h_base, x_offset:x_offset+w_base] = 255
                
                # Blend the images
                mask_inv = cv2.bitwise_not(mask)
                panorama_part = cv2.bitwise_and(new_panorama, new_panorama, mask=mask)
                warped_part = cv2.bitwise_and(warped_img, warped_img, mask=mask_inv)
                
                # Combine both parts
                new_panorama = cv2.add(panorama_part, warped_part)
                
                # Update panorama
                panorama = new_panorama
                h_base, w_base = panorama.shape[:2]
                
                logger.info(f"Successfully stitched image {idx}, new size: {w_base}x{h_base}")
            except Exception as e:
                logger.error(f"Error in custom stitching for image {idx}: {e}")
                continue
                
            # Force garbage collection
            gc.collect()
        
        return panorama

    def process_videos(self, video_paths, output_path="panorama.jpg", sample_rate=15, batch_size=10):
        """Main method to process videos and create a panorama"""
        start_time = time.time()
        logger.info(f"Starting video processing with sample rate {sample_rate}")
        
        # Extract frames from all videos
        all_frames = self.extract_frames_parallel(video_paths, sample_rate)
        
        if len(all_frames) == 0:
            logger.error("No frames were extracted from videos")
            return None
        
        logger.info(f"Extracted {len(all_frames)} frames total")
        
        # Stitch frames in batches
        panorama = self.batch_stitch(all_frames, batch_size)
        
        if panorama is None:
            logger.error("Failed to create panorama")
            return None
        
        # Save the final panorama
        logger.info(f"Saving final panorama to {output_path}")
        cv2.imwrite(output_path, panorama)
        
        # Report processing time
        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        
        return output_path


def cleanup_temp_files(temp_dir):
    """Clean up temporary files"""
    logger.info(f"Cleaning up temporary files in {temp_dir}")
    # Remove all jpg files in temp directory
    for file_path in glob.glob(os.path.join(temp_dir, "*.jpg")):
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Create ultra-high-resolution images from multiple videos")
    
    parser.add_argument("--videos", nargs="+", required=True,
                      help="List of video files to process")
    parser.add_argument("--output", default="panorama.jpg",
                      help="Output filename for the panorama")
    parser.add_argument("--sample-rate", type=int, default=15,
                      help="Frame sampling rate (default: 15)")
    parser.add_argument("--max-resolution", type=int, default=1200,
                      help="Maximum resolution for processing (default: 1200)")
    parser.add_argument("--batch-size", type=int, default=10,
                      help="Batch size for stitching (default: 10)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with intermediate outputs")
    parser.add_argument("--no-cleanup", action="store_true",
                      help="Do not clean up temporary files")
    
    args = parser.parse_args()
    
    # Validate video paths
    for video_path in args.videos:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output) or "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create temp directory
    temp_dir = "temp_frames"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    try:
        # Initialize stitcher
        stitcher = MultiVideoStitcher(
            output_dir=output_dir,
            temp_dir=temp_dir,
            max_resolution=args.max_resolution,
            debug=args.debug
        )
        
        # Process videos
        result = stitcher.process_videos(
            args.videos,
            args.output,
            args.sample_rate,
            args.batch_size
        )
        
        if result is None:
            logger.error("Video processing failed")
            return 1
        
        logger.info(f"Successfully created panorama: {result}")
        
        # Clean up temp files if requested
        if not args.no_cleanup:
            cleanup_temp_files(temp_dir)
        
        return 0
    
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
