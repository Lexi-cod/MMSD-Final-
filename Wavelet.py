#!/usr/bin/env python3
# Wavelet Transform Processor for Panorama Stitching
# Optimizes images with wavelet-based compression before feature matching

import numpy as np
import cv2
import os
import logging
import time
from tqdm import tqdm
import argparse
import sys
from joblib import Parallel, delayed
import multiprocessing
import pywt  # PyWavelets library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class WaveletFrameProcessor:
    """Processes video frames with wavelet transforms to optimize for panorama stitching"""
    
    def __init__(self, output_dir="wavelet_frames", wavelet_type="bior4.4", 
                 decomposition_level=2, threshold_factor=3.0, 
                 mode="soft", transform_type="swt"):
        """
        Initialize Wavelet frame processor
        
        Parameters:
        - output_dir: Directory to store processed frames
        - wavelet_type: Type of wavelet to use (e.g., 'bior4.4', 'db4', 'sym4')
        - decomposition_level: Number of wavelet decomposition levels
        - threshold_factor: Factor used for thresholding wavelet coefficients
        - mode: Thresholding mode - 'soft' or 'hard'
        - transform_type: Type of wavelet transform - 'dwt' (regular), 'swt' (stationary/undecimated)
        """
        self.output_dir = output_dir
        self.wavelet_type = wavelet_type
        self.decomposition_level = decomposition_level
        self.threshold_factor = threshold_factor
        self.mode = mode
        self.transform_type = transform_type
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized Wavelet processor with type={wavelet_type}, "
                   f"level={decomposition_level}, transform={transform_type}")
        
        # Verify PyWavelets supports the selected wavelet
        wavelet_list = pywt.wavelist()
        if wavelet_type not in wavelet_list:
            logger.warning(f"Wavelet type '{wavelet_type}' not found in PyWavelets. "
                          f"Using default 'bior4.4' instead.")
            self.wavelet_type = "bior4.4"
    
    def threshold_coefficients_dwt(self, coeffs, threshold_factor, mode):
        """Apply thresholding to DWT wavelet coefficients"""
        # For DWT, first element is approximation coefficients
        result = [coeffs[0]]  # Keep approximation as is
        
        # Process each detail level
        for i in range(1, len(coeffs)):
            # Each element is a tuple of (horizontal, vertical, diagonal) coefficients
            h, v, d = coeffs[i]
            
            # Calculate threshold based on coefficient statistics
            h_thresh = threshold_factor * np.std(h) / np.sqrt(np.log2(1 + h.size))
            v_thresh = threshold_factor * np.std(v) / np.sqrt(np.log2(1 + v.size))
            d_thresh = threshold_factor * np.std(d) / np.sqrt(np.log2(1 + d.size))
            
            # Apply thresholding
            h_threshed = pywt.threshold(h, h_thresh, mode=mode)
            v_threshed = pywt.threshold(v, v_thresh, mode=mode)
            d_threshed = pywt.threshold(d, d_thresh, mode=mode)
            
            result.append((h_threshed, v_threshed, d_threshed))
        
        return result
    
    def threshold_coefficients_swt(self, coeffs, threshold_factor, mode):
        """Apply thresholding to SWT wavelet coefficients"""
        result = []
        
        # In SWT, coeffs is a list where each element is a tuple of (cA, (cH, cV, cD))
        for i in range(len(coeffs)):
            # Get approximation and detail coefficients
            cA, details = coeffs[i]
            cH, cV, cD = details
            
            # Calculate thresholds
            h_thresh = threshold_factor * np.std(cH) / np.sqrt(np.log2(1 + cH.size))
            v_thresh = threshold_factor * np.std(cV) / np.sqrt(np.log2(1 + cV.size))
            d_thresh = threshold_factor * np.std(cD) / np.sqrt(np.log2(1 + cD.size))
            
            # Apply thresholding to detail coefficients
            cH_threshed = pywt.threshold(cH, h_thresh, mode=mode)
            cV_threshed = pywt.threshold(cV, v_thresh, mode=mode)
            cD_threshed = pywt.threshold(cD, d_thresh, mode=mode)
            
            # cA is not thresholded (approximation)
            result.append((cA, (cH_threshed, cV_threshed, cD_threshed)))
        
        return result
    
    def process_channel_dwt(self, channel):
        """Process a single channel with DWT"""
        # Forward DWT transform
        coeffs = pywt.wavedec2(channel, self.wavelet_type, level=self.decomposition_level)
        
        # Threshold coefficients
        thresholded_coeffs = self.threshold_coefficients_dwt(coeffs, self.threshold_factor, self.mode)
        
        # Inverse DWT transform
        reconstructed = pywt.waverec2(thresholded_coeffs, self.wavelet_type)
        
        # Handle potential size differences (wavelet transform can change dimensions)
        if reconstructed.shape != channel.shape:
            reconstructed = reconstructed[:channel.shape[0], :channel.shape[1]]
        
        return np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    def process_channel_swt(self, channel):
        """Process a single channel with SWT (Stationary Wavelet Transform)"""
        # Ensure dimensions are large enough for SWT
        original_shape = channel.shape
        
        # SWT requires dimensions to be multiple of 2^level
        min_size = 2 ** self.decomposition_level
        padded_h = ((original_shape[0] + min_size - 1) // min_size) * min_size
        padded_w = ((original_shape[1] + min_size - 1) // min_size) * min_size
        
        # Pad the channel if needed
        if padded_h > original_shape[0] or padded_w > original_shape[1]:
            padded = np.zeros((padded_h, padded_w), dtype=channel.dtype)
            padded[:original_shape[0], :original_shape[1]] = channel
            channel = padded
        
        # Forward SWT transform
        coeffs = pywt.swt2(channel, self.wavelet_type, level=self.decomposition_level)
        
        # Threshold coefficients
        thresholded_coeffs = self.threshold_coefficients_swt(coeffs, self.threshold_factor, self.mode)
        
        # Inverse SWT transform
        reconstructed = pywt.iswt2(thresholded_coeffs, self.wavelet_type)
        
        # Crop back to original size
        reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
        
        return np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    def process_image(self, img):
        """Process a single image with wavelet transform"""
        # Convert to YCrCb color space for better compression
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Process each channel with appropriate wavelet transform
        if self.transform_type == 'dwt':
            y_processed = self.process_channel_dwt(y)
            cr_processed = self.process_channel_dwt(cr)
            cb_processed = self.process_channel_dwt(cb)
        else:  # 'swt'
            y_processed = self.process_channel_swt(y)
            cr_processed = self.process_channel_swt(cr)
            cb_processed = self.process_channel_swt(cb)
        
        # Merge channels and convert back to BGR
        ycrcb_processed = cv2.merge([y_processed, cr_processed, cb_processed])
        bgr_processed = cv2.cvtColor(ycrcb_processed, cv2.COLOR_YCrCb2BGR)
        
        return bgr_processed
    
    def extract_and_process_frames(self, video_path, sample_rate=3):
        """Extract frames from video and process them with wavelets"""
        logger.info(f"Processing frames from {video_path} with sample rate {sample_rate}")
        
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
        
        # Define function to extract and process a single frame
        def extract_and_process_single_frame(frame_idx, video_path, output_dir, video_name):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Process the frame with wavelet transform
                processed_frame = self.process_image(frame)
                
                # Save the processed frame
                frame_path = os.path.join(output_dir, f"{video_name}_wavelet_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_path, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return frame_path
            return None
        
        # Use parallel processing
        num_cores = multiprocessing.cpu_count()
        logger.info(f"Using {num_cores} cores for parallel frame processing")
        
        processed_frames = Parallel(n_jobs=num_cores)(
            delayed(extract_and_process_single_frame)(
                frame_idx, video_path, self.output_dir, video_name
            ) for frame_idx in tqdm(frames_to_extract, desc=f"Processing frames from {video_name}")
        )
        
        # Filter out None values (failed extractions)
        processed_frames = [f for f in processed_frames if f is not None]
        
        logger.info(f"Processed {len(processed_frames)} frames from {video_path}")
        return processed_frames
    
    def process_videos(self, video_paths, sample_rate=3):
        """Process multiple videos and return list of processed frame paths"""
        start_time = time.time()
        processed_frames = []
        
        for video_path in video_paths:
            frames = self.extract_and_process_frames(video_path, sample_rate)
            processed_frames.extend(frames)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")
        logger.info(f"Total processed frames: {len(processed_frames)}")
        
        return processed_frames

def main():
    parser = argparse.ArgumentParser(
        description="Process video frames with wavelet transforms for panorama stitching")
    
    parser.add_argument("--videos", nargs="+", required=True,
                      help="List of video files to process")
    parser.add_argument("--output-dir", default="wavelet_frames",
                      help="Output directory for processed frames")
    parser.add_argument("--sample-rate", type=int, default=3,
                      help="Frame sampling rate (default: 3)")
    parser.add_argument("--wavelet", default="bior4.4",
                      help="Wavelet type to use (default: bior4.4)")
    parser.add_argument("--level", type=int, default=2,
                      help="Wavelet decomposition level (default: 2)")
    parser.add_argument("--threshold", type=float, default=3.0,
                      help="Threshold factor for coefficient filtering (default: 3.0)")
    parser.add_argument("--mode", default="soft", choices=["soft", "hard"],
                      help="Thresholding mode (default: soft)")
    parser.add_argument("--transform", default="swt", choices=["dwt", "swt"],
                      help="Wavelet transform type (default: swt)")
    
    args = parser.parse_args()
    
    # Validate video paths
    for video_path in args.videos:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1
    
    try:
        # Verify PyWavelets is installed
        import pywt
    except ImportError:
        logger.error("PyWavelets library not found. Please install with: pip install pywavelets")
        return 1
    
    # Create frame processor
    processor = WaveletFrameProcessor(
        output_dir=args.output_dir,
        wavelet_type=args.wavelet,
        decomposition_level=args.level,
        threshold_factor=args.threshold,
        mode=args.mode,
        transform_type=args.transform
    )
    
    try:
        # Process videos
        processed_frames = processor.process_videos(args.videos, args.sample_rate)
        
        # Write frame list to file for later use with panorama stitcher
        frames_list_path = os.path.join(args.output_dir, "processed_frames_list.txt")
        with open(frames_list_path, 'w') as f:
            f.write('\n'.join(processed_frames))
        
        logger.info(f"Processed frame list written to: {frames_list_path}")
        logger.info("You can now use these frames with your panorama stitcher")
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
