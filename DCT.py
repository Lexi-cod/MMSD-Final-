#!/usr/bin/env python3
# DCT Quantization Processor for Panorama Stitching
# Optimizes images with compression before feature matching

import numpy as np
import cv2
import os
import logging
import time
from tqdm import tqdm
from scipy.fftpack import dct, idct
import argparse
import sys
from joblib import Parallel, delayed
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class DCTFrameProcessor:
    """Processes video frames with DCT quantization to optimize for panorama stitching"""
    
    def __init__(self, output_dir="dct_frames", block_size=8, quality=75):
        """
        Initialize DCT frame processor
        
        Parameters:
        - output_dir: Directory to store processed frames
        - block_size: Size of DCT blocks (typically 8x8)
        - quality: Quality factor (0-100), higher means less compression
        """
        self.output_dir = output_dir
        self.block_size = block_size
        self.quality = quality
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize quantization tables based on quality
        self.init_quantization_tables(quality)
        
        logger.info(f"Initialized DCT processor with quality={quality}, block_size={block_size}")
    
    def init_quantization_tables(self, quality):
        """Initialize standard JPEG-like quantization tables adjusted by quality"""
        # Standard JPEG luminance quantization table
        self.y_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Standard JPEG chrominance quantization table
        self.c_table = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=np.float32)
        
        # Adjust tables based on quality factor
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        
        scale = scale / 100.0
        
        # Scale quantization tables
        self.y_table = np.clip(np.round(self.y_table * scale), 1, 255)
        self.c_table = np.clip(np.round(self.c_table * scale), 1, 255)
    
    def dct_quantize_block(self, block, is_luma=True):
        """Apply DCT and quantization to a single block"""
        # Apply DCT
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
        
        # Quantize
        if is_luma:
            quantized = np.round(dct_block / self.y_table)
        else:
            quantized = np.round(dct_block / self.c_table)
            
        return quantized
    
    def idct_dequantize_block(self, block, is_luma=True):
        """Apply inverse quantization and IDCT to a single block"""
        # Dequantize
        if is_luma:
            dequantized = block * self.y_table
        else:
            dequantized = block * self.c_table
            
        # Apply inverse DCT
        idct_block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho')
            
        return idct_block
    
    def process_image(self, img):
        """Process a single image with DCT quantization"""
        # Convert to YCrCb color space (similar to YUV used in JPEG)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Pad dimensions to be multiple of block_size
        h, w = y.shape
        h_pad = self.block_size - (h % self.block_size) if h % self.block_size != 0 else 0
        w_pad = self.block_size - (w % self.block_size) if w % self.block_size != 0 else 0
        
        # Add padding if needed
        if h_pad > 0 or w_pad > 0:
            y = np.pad(y, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=0)
            cr = np.pad(cr, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=128)
            cb = np.pad(cb, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=128)
        
        h_padded, w_padded = y.shape
        
        # Create output arrays
        y_out = np.zeros_like(y, dtype=np.float32)
        cr_out = np.zeros_like(cr, dtype=np.float32)
        cb_out = np.zeros_like(cb, dtype=np.float32)
        
        # Process blocks
        for i in range(0, h_padded, self.block_size):
            for j in range(0, w_padded, self.block_size):
                # Extract blocks
                y_block = y[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                cr_block = cr[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                cb_block = cb[i:i+self.block_size, j:j+self.block_size].astype(np.float32)
                
                # Center values around zero (similar to JPEG)
                y_block -= 128
                cr_block -= 128
                cb_block -= 128
                
                # Apply DCT and quantize
                y_q = self.dct_quantize_block(y_block, is_luma=True)
                cr_q = self.dct_quantize_block(cr_block, is_luma=False)
                cb_q = self.dct_quantize_block(cb_block, is_luma=False)
                
                # Apply inverse quantization and IDCT
                y_block = self.idct_dequantize_block(y_q, is_luma=True)
                cr_block = self.idct_dequantize_block(cr_q, is_luma=False)
                cb_block = self.idct_dequantize_block(cb_q, is_luma=False)
                
                # Restore original range
                y_block += 128
                cr_block += 128
                cb_block += 128
                
                # Clip values to valid range
                y_block = np.clip(y_block, 0, 255)
                cr_block = np.clip(cr_block, 0, 255)
                cb_block = np.clip(cb_block, 0, 255)
                
                # Store in output arrays
                y_out[i:i+self.block_size, j:j+self.block_size] = y_block
                cr_out[i:i+self.block_size, j:j+self.block_size] = cr_block
                cb_out[i:i+self.block_size, j:j+self.block_size] = cb_block
        
        # Crop to original size
        y_out = y_out[:h, :w]
        cr_out = cr_out[:h, :w]
        cb_out = cb_out[:h, :w]
        
        # Merge channels and convert back to BGR
        ycrcb_out = cv2.merge([y_out.astype(np.uint8), 
                               cr_out.astype(np.uint8), 
                               cb_out.astype(np.uint8)])
        bgr_out = cv2.cvtColor(ycrcb_out, cv2.COLOR_YCrCb2BGR)
        
        return bgr_out
    
    def extract_and_process_frames(self, video_path, sample_rate=3):
        """Extract frames from video and process them with DCT quantization"""
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
                # Process the frame with DCT quantization
                processed_frame = self.process_image(frame)
                
                # Save the processed frame
                frame_path = os.path.join(output_dir, f"{video_name}_dct_{frame_idx:04d}.jpg")
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
        description="Process video frames with DCT quantization for panorama stitching")
    
    parser.add_argument("--videos", nargs="+", required=True,
                      help="List of video files to process")
    parser.add_argument("--output-dir", default="dct_frames",
                      help="Output directory for processed frames")
    parser.add_argument("--sample-rate", type=int, default=3,
                      help="Frame sampling rate (default: 3)")
    parser.add_argument("--quality", type=int, default=75,
                      help="Quality factor (0-100, default: 75)")
    parser.add_argument("--block-size", type=int, default=8,
                      help="DCT block size (default: 8)")
    
    args = parser.parse_args()
    
    # Validate video paths
    for video_path in args.videos:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return 1
    
    # Create frame processor
    processor = DCTFrameProcessor(
        output_dir=args.output_dir,
        block_size=args.block_size,
        quality=args.quality
    )
    
    try:
        # Process videos
        processed_frames = processor.process_videos(args.videos, args.sample_rate)
        
        # Write frame list to file for later use
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
