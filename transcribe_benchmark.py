#!/usr/bin/env python
"""
Benchmark script for measuring transcription speed on a single audio file.

This script processes a single audio file using the Faster Whisper model
and measures the transcription speed for a single pass.
"""

import argparse
import time
import numpy as np
import torch
from faster_whisper import WhisperModel
import soundfile as sf
from datetime import datetime
import datetime as dt

def main():
    # Command line arguments
    parser = argparse.ArgumentParser()
    # Model selection
    parser.add_argument("--model", default="distil-medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", 
                                 "distil-medium", "distil-large", "distil-large-v2", "distil-large-v3"])
    # Input file
    parser.add_argument("--input", required=True, help="Path to audio file (mp3, wav, etc.)")
    # Language options
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    # Performance options
    parser.add_argument("--compute_type", default="auto", 
                        choices=["auto", "int8", "int8_float16", "int16", "float16", "float32"],
                        help="Compute type for model inference")
    parser.add_argument("--cpu_threads", default=8, type=int,
                        help="Number of CPU threads to use for processing")
    # Output options
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information during processing")
    
    args = parser.parse_args()
    
    # Load the audio file
    print(f"Loading audio file: {args.input}")
    try:
        audio, sample_rate = sf.read(args.input)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    # Resample to 16kHz if needed (Whisper expects 16kHz audio)
    if sample_rate != 16000:
        print(f"Warning: Audio sample rate is {sample_rate}Hz, Whisper expects 16kHz audio.")
        print("Consider resampling your audio file to 16kHz for best results.")
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        print("Converting stereo audio to mono")
        audio = audio.mean(axis=1)
    
    # Convert to float32 and normalize
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if audio is not already normalized
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    
    # Select model variant
    model_name = args.model
    if not args.non_english and not model_name.endswith(".en"):
        model_name = model_name + ".en"  # Use English-specific model for better accuracy
    
    # Determine compute type - use int8 for distilled models by default
    compute_type = args.compute_type
    if compute_type == "auto" and "distil" in model_name:
        compute_type = "int8"
    
    print(f"Loading {model_name} model...")
    model_start_time = time.time()
    # Initialize the model
    audio_model = WhisperModel(
        model_name, 
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        compute_type=compute_type,
        cpu_threads=args.cpu_threads
    )
    model_load_time = time.time() - model_start_time
    print(f"Model loaded in {model_load_time:.2f} seconds")
    
    # Get audio file info
    audio_duration = len(audio) / sample_rate
    print(f"Audio file duration: {audio_duration:.2f} seconds")
    
    # Print system info
    device = 'CUDA' if torch.cuda.is_available() else 'CPU'
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Perform transcription and measure time
    print("\nStarting transcription...")
    start_time = time.time()
    segments, info = audio_model.transcribe(
        audio, 
        beam_size=12,  # Larger beam size for better accuracy
        language='en' if not args.non_english else None,
        vad_filter=True
    )
    
    # Collect results
    segments_list = list(segments)  # Convert generator to list
    transcription = ''.join(segment.text for segment in segments_list)
    end_time = time.time()
    transcription_time = end_time - start_time
    
    # Calculate processing metrics
    realtime_factor = transcription_time / audio_duration
    processing_speed = audio_duration / transcription_time
    
    # Print results
    print("\n" + "="*50)
    print("TRANSCRIPTION BENCHMARK RESULTS")
    print("="*50)
    print(f"Audio duration:       {audio_duration:.2f} seconds")
    print(f"Transcription time:   {transcription_time:.2f} seconds")
    print(f"Realtime factor:      {realtime_factor:.2f}x")
    print(f"Processing speed:     {processing_speed:.2f}x realtime")
    print("="*50)
    
    if args.verbose:
        print("\nDetailed Information:")
        print(f"Model:                {model_name}")
        print(f"Compute type:         {compute_type}")
        print(f"Device:               {device}")
        print(f"Language:             {'English' if not args.non_english else 'Auto-detect'}")
        print(f"Detected language:    {info.language} ({info.language_probability:.2f})")
        print(f"Number of segments:   {len(segments_list)}")
        
        print("\nSegment Details:")
        for i, segment in enumerate(segments_list):
            print(f"Segment {i+1}/{len(segments_list)}: {segment.start:.2f}s → {segment.end:.2f}s | {segment.text}")
    
    print("\nTranscription:")
    print(transcription)
    
    # Save transcription to file with the same name as the input but with .txt extension
    output_file = args.input.rsplit('.', 1)[0] + '.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcription)
    print(f"\nTranscription saved to: {output_file}")

if __name__ == "__main__":
    main()
