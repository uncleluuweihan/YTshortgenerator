#!/usr/bin/env python3
"""
Uniswap v4 YouTube Shorts Generator (OpenCV Version)

This script automates the generation of YouTube Shorts content focused on Uniswap v4.
It processes a text file input to create AI-generated video and voiceover audio.

Requirements:
- Python 3.8+
- OpenAI API key (for GPT-4 and DALL-E 3)
- ElevenLabs API key (for voice synthesis)
- OpenCV, ffmpeg-python, and Pillow libraries
- FFMPEG installed
"""

import os
import sys
import json
import time
import argparse
import tempfile
import textwrap
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import ffmpeg
import requests
from PIL import Image, ImageDraw, ImageFont
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# Constants
DEFAULT_VIDEO_SIZE = (1080, 1920)  # Portrait for YouTube Shorts
DEFAULT_FONT_SIZE = 40
DEFAULT_DURATION = 58  # YouTube Shorts are up to 60 seconds

class UniswapShortsGenerator:
    """Main class for generating Uniswap v4 YouTube Shorts"""
    
    def __init__(self, 
                 input_file: str, 
                 output_dir: str = "output",
                 duration: int = DEFAULT_DURATION,
                 voice_id: str = "21m00Tcm4TlvDq8ikWAM"):  # Default ElevenLabs voice
        """
        Initialize the shorts generator
        
        Args:
            input_file: Path to the input text file
            output_dir: Directory for output files
            duration: Target duration in seconds
            voice_id: ElevenLabs voice ID
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.duration = duration
        self.voice_id = voice_id
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State variables
        self.script = None
        self.segments = None
        self.images = []
        self.audio_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Try to load font, fallback to default if not found
        try:
            # For Windows
            self.font = ImageFont.truetype("arial.ttf", DEFAULT_FONT_SIZE)
        except IOError:
            try:
                # For macOS
                self.font = ImageFont.truetype("/Library/Fonts/Arial.ttf", DEFAULT_FONT_SIZE)
            except IOError:
                # Fallback to default
                self.font = ImageFont.load_default()
    
    def read_input_file(self) -> str:
        """Read the input file content"""
        with open(self.input_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    def process_script(self):
        """Process the input script and break it into segments"""
        raw_text = self.read_input_file()
        
        # Use GPT-4 to improve and segment the script
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in creating engaging, educational content about Uniswap v4 for YouTube Shorts. Your task is to take the provided text and transform it into a well-structured script divided into 4-7 segments. Each segment should be concise (no more than 2-3 sentences) and build on the previous one. Focus on making the content engaging, accurate, and easy to understand for an audience interested in DeFi and crypto."},
                {"role": "user", "content": f"Here's the text I want to transform into a YouTube Shorts script about Uniswap v4:\n\n{raw_text}\n\nPlease create a segmented script optimized for YouTube Shorts (under 60 seconds total). Format your response as a JSON array where each element is an object with 'text' and 'image_prompt' fields. The 'image_prompt' should be a detailed prompt for generating an image that illustrates that segment."}
            ]
        )
        
        # Parse the response
        response_text = response.choices[0].message.content
        try:
            # Extract JSON if it's embedded in the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
                self.segments = json.loads(json_str)
            else:
                # Try to find and parse JSON directly
                json_start = response_text.find('[')
                json_end = response_text.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    self.segments = json.loads(json_str)
                else:
                    raise ValueError("Couldn't extract JSON from the API response")
        except json.JSONDecodeError:
            print("Error parsing JSON response. Falling back to manual segmentation.")
            # Basic fallback segmentation
            paragraphs = raw_text.split('\n\n')
            self.segments = []
            for p in paragraphs:
                if not p.strip():
                    continue
                self.segments.append({
                    "text": p,
                    "image_prompt": f"An illustrative image for Uniswap v4 content: {p[:100]}..."
                })
        
        # Ensure we don't have too many segments
        if len(self.segments) > 7:
            self.segments = self.segments[:7]
        
        print(f"Processed script into {len(self.segments)} segments")
    
    def generate_images(self):
        """Generate images for each segment using DALL-E 3"""
        for i, segment in enumerate(self.segments):
            print(f"Generating image {i+1}/{len(self.segments)}...")
            
            prompt = f"Create a high-quality, eye-catching image for a YouTube Short about Uniswap v4. {segment['image_prompt']} Style: Modern, digital, financial technology, cryptocurrency theme. Include Uniswap branding elements or colors."
            
            try:
                response = openai.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1792",  # Closest to YouTube Shorts aspect ratio
                    quality="standard",
                    n=1
                )
                
                image_url = response.data[0].url
                
                # Download the image
                img_response = requests.get(image_url)
                img_path = self.output_dir / f"image_{i+1}.png"
                
                with open(img_path, "wb") as img_file:
                    img_file.write(img_response.content)
                
                self.images.append(str(img_path))
                print(f"Image {i+1} saved to {img_path}")
                
                # Rate limiting to avoid API limits
                time.sleep(2)
                
            except Exception as e:
                print(f"Error generating image {i+1}: {str(e)}")
                # Use a placeholder image
                self.images.append(None)
    
    def generate_voiceover(self):
        """Generate voiceover audio for each segment using ElevenLabs"""
        for i, segment in enumerate(self.segments):
            print(f"Generating audio {i+1}/{len(self.segments)}...")
            
            text = segment['text']
            
            try:
                # Call ElevenLabs API
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": elevenlabs_api_key
                }
                data = {
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
                
                response = requests.post(url, json=data, headers=headers)
                
                if response.status_code == 200:
                    # Save the audio file
                    audio_path = self.output_dir / f"audio_{i+1}.mp3"
                    with open(audio_path, "wb") as audio_file:
                        audio_file.write(response.content)
                    
                    self.audio_files.append(str(audio_path))
                    print(f"Audio {i+1} saved to {audio_path}")
                else:
                    print(f"Error generating audio {i+1}: {response.status_code} {response.text}")
                    self.audio_files.append(None)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating audio {i+1}: {str(e)}")
                self.audio_files.append(None)
    
    def add_text_to_image(self, image_path: str, text: str) -> str:
        """
        Add text overlay to an image
        
        Args:
            image_path: Path to the source image
            text: Text to add to the image
            
        Returns:
            Path to the new image with text overlay
        """
        # Open the image with PIL
        img = Image.open(image_path)
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Calculate text position (bottom of image)
        wrapped_text = textwrap.fill(text, width=30)
        
        # Get text size
        # Use a simpler approach since not all PIL versions have textbbox
        text_lines = wrapped_text.count('\n') + 1
        estimated_text_height = self.font.size * text_lines
        
        # Create semi-transparent background for text
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        text_y_position = img.height - estimated_text_height - 40  # padding
        overlay_draw.rectangle(
            [(0, text_y_position - 10), (img.width, img.height)],
            fill=(0, 0, 0, 128)
        )
        
        # Apply the overlay
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')  # Convert back to RGB for jpg support
        
        # Add text
        draw = ImageDraw.Draw(img)
        draw.multiline_text(
            (img.width // 2, text_y_position),
            wrapped_text,
            font=self.font,
            fill=(255, 255, 255),
            align="center",
            anchor="ma"  # middle alignment if supported
        )
        
        # Save the modified image
        output_path = self.temp_dir / f"text_{Path(image_path).name}"
        img.save(output_path)
        
        return str(output_path)
    
    def create_intro_frame(self) -> str:
        """
        Create an intro frame with title
        
        Returns:
            Path to the intro frame image
        """
        # Create a blank image (Uniswap pink background)
        img = Image.new('RGB', DEFAULT_VIDEO_SIZE, (255, 51, 153))
        draw = ImageDraw.Draw(img)
        
        # Add title text
        title = "Uniswap v4 Explained"
        
        # Try to use a larger font for the title
        try:
            title_font = ImageFont.truetype("arial.ttf", 60)
        except IOError:
            try:
                title_font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 60)
            except IOError:
                title_font = self.font
        
        # Get text size for centering
        # Simple approach for centering
        draw.text(
            (DEFAULT_VIDEO_SIZE[0] // 2, DEFAULT_VIDEO_SIZE[1] // 2),
            title,
            font=title_font,
            fill=(255, 255, 255),
            align="center",
            anchor="mm"  # middle alignment if supported
        )
        
        # Save the intro frame
        output_path = self.temp_dir / "intro_frame.png"
        img.save(output_path)
        
        return str(output_path)
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file using ffmpeg
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        probe = ffmpeg.probe(audio_path)
        audio_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'audio')
        return float(audio_info['duration'])
    
    def create_video(self):
        """Create the final video by combining images and audio using OpenCV and ffmpeg"""
        # Check if we have any valid segments
        valid_segments = [(i, img, audio) for i, (img, audio) in enumerate(zip(self.images, self.audio_files))
                          if img is not None and audio is not None]
        
        if not valid_segments:
            print("No valid segments found. Cannot create video.")
            return None
        
        segment_clips = []
        total_duration = 0
        
        # Process each segment
        for i, img_path, audio_path in valid_segments:
            # Get audio duration
            audio_duration = self.get_audio_duration(audio_path)
            
            # Add text overlay to image
            text = self.segments[i]['text']
            img_with_text = self.add_text_to_image(img_path, text)
            
            # Create a temporary video segment with the image and audio
            segment_output = self.temp_dir / f"segment_{i}.mp4"
            
            # Apply zoom effect by creating a video with OpenCV
            img = cv2.imread(img_with_text)
            height, width, _ = img.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            writer = cv2.VideoWriter(str(segment_output), fourcc, fps, (width, height))
            
            # Calculate total frames
            total_frames = int(audio_duration * fps)
            
            # Apply zoom effect
            for frame_num in range(total_frames):
                progress = frame_num / total_frames
                zoom_factor = 1 + 0.05 * progress  # 5% zoom
                
                # Create zoomed frame
                zoomed_size = int(width * zoom_factor), int(height * zoom_factor)
                zoomed_img = cv2.resize(img, zoomed_size)
                
                # Calculate crop to maintain original size
                start_x = (zoomed_size[0] - width) // 2
                start_y = (zoomed_size[1] - height) // 2
                
                # Crop the zoomed image
                cropped = zoomed_img[start_y:start_y+height, start_x:start_x+width]
                
                # Write the frame
                writer.write(cropped)
            
            writer.release()
            
            # Add audio to the segment using ffmpeg
            segment_with_audio = self.temp_dir / f"segment_{i}_with_audio.mp4"
            
            # Use ffmpeg to add audio
            try:
                (
                    ffmpeg
                    .input(str(segment_output))
                    .input(audio_path)
                    .output(str(segment_with_audio), shortest=None, vcodec='copy')
                    .overwrite_output()
                    .run(quiet=True)
                )
            except ffmpeg.Error as e:
                print(f"Error adding audio to segment {i}: {e.stderr.decode()}")
                continue
            
            segment_clips.append(str(segment_with_audio))
            total_duration += audio_duration
        
        # Create intro if there's time
        if total_duration < self.duration - 3:
            intro_img = self.create_intro_frame()
            intro_output = self.temp_dir / "intro.mp4"
            
            # Create intro video (3 seconds)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            img = cv2.imread(intro_img)
            height, width, _ = img.shape
            
            writer = cv2.VideoWriter(str(intro_output), fourcc, fps, (width, height))
            
            # Write the same frame for 3 seconds
            for _ in range(3 * fps):
                writer.write(img)
            
            writer.release()
            
            segment_clips.insert(0, str(intro_output))
        
        # Concatenate all segments using ffmpeg
        if not segment_clips:
            print("No valid segments to concatenate. Video creation failed.")
            return None
        
        # Create a list file for ffmpeg concat
        concat_file = self.temp_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for clip in segment_clips:
                f.write(f"file '{clip}'\n")
        
        # Output final video
        output_path = self.output_dir / "uniswap_short.mp4"
        
        # Concatenate videos
        try:
            subprocess.run([
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path)
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating videos: {e}")
            return None
        
        print(f"Video created successfully: {output_path}")
        return output_path
    
    def generate(self) -> str:
        """Generate the complete YouTube Short"""
        self.process_script()
        self.generate_images()
        self.generate_voiceover()
        return self.create_video()


def main():
    """Main function to parse arguments and run the generator"""
    parser = argparse.ArgumentParser(description="Generate YouTube Shorts about Uniswap v4")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-d", "--duration", type=int, default=DEFAULT_DURATION, help="Target duration in seconds")
    parser.add_argument("-v", "--voice", default="21m00Tcm4TlvDq8ikWAM", help="ElevenLabs voice ID")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Check for API keys
    if not openai.api_key:
        print("Error: OpenAI API key not found. Set it as OPENAI_API_KEY environment variable or in .env file.")
        sys.exit(1)
    
    if not elevenlabs_api_key:
        print("Error: ElevenLabs API key not found. Set it as ELEVENLABS_API_KEY environment variable or in .env file.")
        sys.exit(1)
    
    # Generate the short
    generator = UniswapShortsGenerator(
        input_file=args.input_file,
        output_dir=args.output,
        duration=args.duration,
        voice_id=args.voice
    )
    
    output_path = generator.generate()
    print(f"✅ YouTube Short successfully generated at {output_path}")


if __name__ == "__main__":
    main()