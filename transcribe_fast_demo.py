#!/usr/bin/env python
"""
Real-time speech transcription using Faster Whisper model.

This script continuously listens to audio input from the microphone,
transcribes it using the Faster Whisper model, and displays the transcription
in real-time. It supports various models, including distilled models for improved performance.
"""

import argparse
import numpy as np
import speech_recognition as sr  # For microphone access and voice activity detection
import torch
from datetime import datetime, timedelta
import datetime as dt
from queue import Queue  # Thread-safe queue for audio data
from time import sleep
from sys import platform  # Used for platform-specific microphone setup
from threading import Thread
from faster_whisper import WhisperModel  # Faster implementation of Whisper

def main():
    # ========== COMMAND LINE ARGUMENT SETUP ==========
    parser = argparse.ArgumentParser()
    # Model selection - more options than original Whisper, including distilled models
    parser.add_argument("--model", default="distil-medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", 
                                 "distil-medium", "distil-large", "distil-large-v2", "distil-large-v3"])
    # Language options
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    # Audio detection parameters
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    # Output options
    parser.add_argument("--timestamp", action="store_true",
                        help="Add timestamps to each transcription entry")
    parser.add_argument("--procdur", action="store_true",
                        help="Include processing duration in parenthesis after timestamp")
    
    # Platform-specific microphone configuration
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    elif 'darwin' in platform:  # macOS
        parser.add_argument("--default_microphone", default='Built-in',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # ========== AUDIO PROCESSING SETUP ==========
    # Tracks the last time a recording was retrieved from the queue
    phrase_time = None
    # Thread-safe Queue for passing audio data from recording thread to transcription thread
    data_queue = Queue()
    # SpeechRecognizer handles microphone recording with voice activity detection
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Disable dynamic energy threshold adjustment - keeps our manual threshold setting
    recorder.dynamic_energy_threshold = False

    # ========== MICROPHONE SETUP ==========
    # Platform-specific microphone selection to prevent crashes
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            # Just list available microphones and exit
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            # Search for the specified microphone by name
            source = None
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
            # Fallback to default if specified microphone not found
            if source is None:
                source = sr.Microphone(sample_rate=16000)
    else:
        # On non-Linux platforms, use the default microphone
        source = sr.Microphone(sample_rate=16000)  # 16kHz sample rate is optimal for Whisper

    # ========== MODEL SETUP ==========
    # Select model variant and determine if language-specific model should be used
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"  # Use English-specific model for better accuracy on English speech
    
    # Distilled models can use int8 quantization for better performance
    compute_type = 'int8' if 'distil' in model else 'auto'
    print(f"Loading {model} model...")
    audio_model = WhisperModel(model, 
                               device='cuda' if torch.cuda.is_available() else 'cpu', 
                               compute_type=compute_type,
                               cpu_threads=8)  # Optimize CPU usage with more threads

    # Store parameters for recording and phrase detection
    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    
    # ========== TRANSCRIPTION STORAGE SETUP ==========
    # Initialize empty transcription list
    # If timestamps are enabled, each entry will be a tuple of (timestamp, text, proc_time)
    # Otherwise, it will be a list of text strings
    transcription = []
    if args.timestamp:
        transcription.append((datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S"), "", 0.0))
    else:
        transcription.append("")

    # ========== AUDIO INPUT CALIBRATION ==========
    # Adjust for ambient noise to calibrate the energy threshold
    with source:
        recorder.adjust_for_ambient_noise(source)

    # ========== AUDIO RECORDING CALLBACK ==========
    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        This function is called in a separate thread by SpeechRecognition.
        
        Parameters:
            _: Unused recognizer instance
            audio: An AudioData object containing the recorded bytes
        """
        # Extract raw bytes from the audio data and put it in the thread-safe queue
        data = audio.get_raw_data()
        data_queue.put(data)

    # Start background listening for audio - this runs in a separate thread
    # The record_callback will be called whenever speech is detected and recorded
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("Model loaded.\n")

    # ========== TRANSCRIPTION PROCESSING FUNCTION ==========
    def transcribe_audio():
        """
        Main transcription function that runs in a separate thread.
        Continuously pulls audio data from the queue and processes it.
        """
        nonlocal phrase_time
        while True:
            now = datetime.now(dt.UTC)
            # Check if we have audio data to process
            if not data_queue.empty():
                phrase_complete = False
                # Determine if this is a new phrase based on timeout
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # Update the last time we received audio
                phrase_time = now

                # ===== AUDIO DATA PREPARATION =====
                # Combine all audio chunks currently in the queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                # Convert raw audio bytes to numpy array that Whisper can process
                # 1. Convert from int16 PCM audio to float32
                # 2. Scale to range [-1.0, 1.0] by dividing by 32768.0 (2^15)
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # ===== CONTEXT PREPARATION =====
                # Build initial prompt from previous transcriptions to provide context
                initial_prompt = ""
                if transcription and len(transcription) > 0:
                    # Use up to the last 3 entries as context for the model
                    context_entries = transcription[-3:] if len(transcription) >= 3 else transcription
                    if args.timestamp:
                        # Extract just the text from timestamp tuples
                        initial_prompt = " ".join([entry[1] for entry in context_entries if entry[1]])
                    else:
                        initial_prompt = " ".join([entry for entry in context_entries if entry])
                
                # ===== TRANSCRIPTION =====
                # Process audio with Faster Whisper model and measure performance
                start_time = datetime.now(dt.UTC)
                # Note: Faster Whisper returns segments rather than a single text string
                segments, info = audio_model.transcribe(
                    audio_np, 
                    beam_size=12,  # Larger beam size for better accuracy
                    initial_prompt=initial_prompt  # Provide context from previous transcriptions
                )
                proc_time = (datetime.now(dt.UTC) - start_time).total_seconds()
                
                # Join all segments into a single text string
                text = ''.join(segment.text for segment in segments).strip()

                # ===== RESULT HANDLING =====
                # Handle output differently based on whether this is a new phrase or continuation
                if args.timestamp:
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    if phrase_complete:
                        # Add a new entry for a new phrase
                        transcription.append((timestamp, text, proc_time))
                    else:
                        # Update the existing entry with new transcription
                        transcription[-1] = (timestamp, text, proc_time)
                    
                    # Format display lines with timestamps and optional processing duration
                    display_lines = []
                    for line in transcription:
                        ts, txt, pt = line
                        if args.procdur:
                            display_lines.append(f"{ts} ({pt:.2f}s) | {txt}")
                        else:
                            display_lines.append(f"{ts} | {txt}")
                else:
                    if phrase_complete:
                        # Add a new entry for a new phrase
                        transcription.append(text)
                    else:
                        # Update the existing entry with new transcription
                        transcription[-1] = text
                    display_lines = transcription

                # ===== DISPLAY OUTPUT =====
                # Display the full transcription every time (not just the latest part)
                for line in display_lines:
                    print(line)
                print('', flush=True)  # Add blank line and ensure output is visible
            else:
                # No audio data to process, sleep briefly to avoid CPU spinning
                sleep(0.02)

    # ========== THREAD MANAGEMENT ==========
    # Start transcription in a separate thread
    transcribe_thread = Thread(target=transcribe_audio)
    transcribe_thread.start()

    # Main thread just waits for keyboard interrupt (Ctrl+C)
    try:
        while True:
            sleep(0.02)
    except KeyboardInterrupt:
        # No explicit cleanup like in the original Whisper script,
        # but we do join the thread to wait for it to finish
        transcribe_thread.join()

    # ========== FINAL OUTPUT ==========
    # Print the complete transcription at the end
    print("\n\nTranscription:")
    if args.timestamp:
        for line in transcription:
            timestamp, text, proc_time = line
            if args.procdur:
                print(f"{timestamp} ({proc_time:.2f}s) | {text}")
            else:
                print(f"{timestamp} | {text}")
    else:
        for line in transcription:
            print(line)

# ========== SCRIPT ENTRY POINT ==========
if __name__ == "__main__":
    main()

