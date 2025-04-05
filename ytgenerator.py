#!/usr/bin/env python3
"""
Uniswap v4 YouTube Shorts Generator

This script automates the generation of YouTube Shorts content focused on Uniswap v4.
It processes a text file input to create AI-generated video and voiceover audio.

Requirements:
- Python 3.8+
- OpenAI API key (for GPT-4 and DALL-E 3)
- ElevenLabs API key (for voice synthesis)
- MoviePy library
- FFMPEG installed
"""

import os
import sys
import json
import time
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
import openai
from dotenv import load_dotenv
from moviepy.editor import (
    TextClip, ImageClip, AudioFileClip, CompositeVideoClip, 
    concatenate_videoclips, vfx
)
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# Constants
DEFAULT_VIDEO_SIZE = (1080, 1920)  # Portrait for YouTube Shorts
DEFAULT_FONT_SIZE = 40
DEFAULT_FONT = "Arial"
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
    
    def create_video(self):
        """Create the final video by combining images and audio"""
        clips = []
        total_duration = 0
        
        for i, (segment, img_path, audio_path) in enumerate(zip(self.segments, self.images, self.audio_files)):
            if audio_path is None or img_path is None:
                continue
                
            # Load audio and get its duration
            audio_clip = AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            
            # Create image clip
            img_clip = ImageClip(img_path, duration=audio_duration)
            img_clip = img_clip.resize(height=DEFAULT_VIDEO_SIZE[1])
            img_clip = img_clip.set_position("center")
            
            # Add subtle zoom effect
            img_clip = img_clip.fx(vfx.resize, lambda t: 1 + 0.05 * t / audio_duration)
            
            # Add text overlay
            text = textwrap.fill(segment['text'], width=30)
            txt_clip = TextClip(
                text, 
                fontsize=DEFAULT_FONT_SIZE, 
                font=DEFAULT_FONT, 
                color='white',
                bg_color='rgba(0,0,0,0.5)',
                method='caption',
                size=(DEFAULT_VIDEO_SIZE[0] * 0.9, None)
            )
            txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(audio_duration)
            
            # Combine image and text
            composite = CompositeVideoClip([img_clip, txt_clip])
            composite = composite.set_audio(audio_clip)
            
            clips.append(composite)
            total_duration += audio_duration
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Add intro text if there's time
        if total_duration < self.duration - 3:
            intro_txt = TextClip(
                "Uniswap v4 Explained", 
                fontsize=60, 
                font=DEFAULT_FONT, 
                color='white',
                bg_color='rgba(0,0,0,0.7)',
                size=(DEFAULT_VIDEO_SIZE[0], None)
            )
            intro_txt = intro_txt.set_position('center').set_duration(3)
            intro = CompositeVideoClip([
                ColorClip(DEFAULT_VIDEO_SIZE, color=(255, 51, 153), duration=3),  # Uniswap pink background
                intro_txt
            ])
            final_clip = concatenate_videoclips([intro, final_clip], method="compose")
        
        # Ensure the final video has the correct dimensions
        final_clip = final_clip.resize(DEFAULT_VIDEO_SIZE)
        
        # Write the final video file
        output_path = self.output_dir / "uniswap_short.mp4"
        final_clip.write_videofile(
            str(output_path),
            fps=30,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="ultrafast"  # For faster rendering during development
        )
        
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