#!/usr/bin/env python
"""
Real-time speech transcription using OpenAI's Whisper or Faster Whisper.

This script continuously listens to audio input from the microphone,
transcribes it using either the original Whisper model or Faster Whisper model,
and displays the transcription in real-time.
"""

import argparse
import os
import numpy as np
import speech_recognition as sr
import torch
import threading
from datetime import datetime, timedelta
import datetime as dt
from queue import Queue
from time import sleep
from sys import platform

def main():
    # ========== COMMAND LINE ARGUMENT SETUP ==========
    parser = argparse.ArgumentParser()
    # Model and implementation selection
    parser.add_argument("--fastwhisper", action="store_true",
                        help="Use the Faster Whisper implementation instead of the original Whisper")
    parser.add_argument("--model", default="medium", help="Model to use",
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
    # Performance options (original Whisper only)
    parser.add_argument("--threads", default=4, type=int,
                        help="Number of CPU threads to use for processing (original Whisper only)")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"],
                        help="Device to use for inference (cpu, cuda, or auto for automatic detection)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half-precision floating point for inference (original Whisper only)")
    # Output options
    parser.add_argument("--out", type=str, metavar="FILE",
                        help="Output file to write transcription results")
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
    # The last time a recording was retrieved from the queue - used for phrase timeout detection
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
    
    # Load appropriate model based on implementation choice
    audio_model = None
    device = args.device
    if args.fastwhisper:
        # Import Faster Whisper implementation
        from faster_whisper import WhisperModel
        
        # Distilled models can use int8 quantization for better performance
        compute_type = 'int8' if 'distil' in model else 'float16'
        print(f"Loading Faster Whisper {model} model...")
        
        # Determine compute device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # On Apple Silicon, force CPU with specific optimizations
        if 'darwin' in platform and 'arm' in os.uname().machine:
            device = "cpu"
            
        audio_model = WhisperModel(model, 
                                  device=device,
                                  compute_type=compute_type,
                                  cpu_threads=args.threads,
                                  num_workers=2)  # Use parallel processing
    else:
        # Import original Whisper implementation
        import whisper
        
        # Determine the compute device (GPU or CPU)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Optimize CPU usage if using CPU for inference
        if device == "cpu":
            torch.set_num_threads(args.threads)
            
        # Load the specified Whisper model
        print(f"Loading original Whisper {model} model...")
        audio_model = whisper.load_model(model).to(device)

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
        
    # Setup output file if requested
    out_file = None
    if args.out:
        out_file = open(args.out, "w", encoding="utf-8")

    # ========== AUDIO INPUT CALIBRATION ==========
    # Adjust for ambient noise to calibrate the energy threshold
    with source:
        recorder.adjust_for_ambient_noise(source)

    # ========== AUDIO RECORDING CALLBACK ==========
    def record_callback(_, audio:sr.AudioData) -> None:
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

    # Let user know we're ready
    print(f"Model loaded on {device} device.\n")
    
    # Flag for graceful thread termination
    running = True
    
    # ========== TRANSCRIPTION PROCESSING LOOP ==========
    def transcribe_loop():
        """
        Main transcription loop that runs in a separate thread.
        Continuously pulls audio data from the queue and processes it.
        """
        nonlocal phrase_time
        
        while running:
            try:
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
                    # Process audio with appropriate Whisper model and measure performance
                    start_time = datetime.now(dt.UTC)
                    text = ""
                    
                    if args.fastwhisper:
                        # Faster Whisper returns segments
                        segments, info = audio_model.transcribe(
                            audio_np, 
                            beam_size=5,  # Smaller beam size is faster with minimal accuracy loss
                            initial_prompt=initial_prompt,  # Provide context from previous transcriptions
                            vad_filter=True,  # Filter out non-speech audio
                            vad_parameters=dict(min_silence_duration_ms=500)  # Optimize VAD for real-time
                        )
                        # Join all segments into a single text string
                        text = ''.join(segment.text for segment in segments).strip()
                    else:
                        # Original Whisper returns a dictionary with 'text' key
                        result = audio_model.transcribe(
                            audio_np,
                            fp16=args.fp16 or (device == "cuda"),  # Use FP16 on GPU
                            initial_prompt=initial_prompt  # Provide context from previous transcriptions
                        )
                        text = result['text'].strip()
                        
                    proc_time = (datetime.now(dt.UTC) - start_time).total_seconds()

                    # ===== RESULT HANDLING =====
                    # Handle output differently based on whether this is a new phrase or continuation
                    if phrase_complete:
                        # Start a new transcription entry for a new phrase
                        if args.timestamp:
                            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                            transcription.append((timestamp, text, proc_time))
                            if args.procdur:
                                display_text = f"{timestamp} ({proc_time:.2f}s) | {text}"
                            else:
                                display_text = f"{timestamp} | {text}"
                        else:
                            transcription.append(text)
                            display_text = text
                    else:
                        # Update the current transcription entry (overwrite with new version)
                        if args.timestamp:
                            transcription[-1] = (transcription[-1][0], text, proc_time)
                            if args.procdur:
                                display_text = f"{transcription[-1][0]} ({proc_time:.2f}s) | {text}"
                            else:
                                display_text = f"{transcription[-1][0]} | {text}"
                        else:
                            transcription[-1] = text
                            display_text = text
                    
                    # Write to file if specified (always write updates, not just on phrase completion)
                    if out_file:
                        if args.timestamp and args.procdur:
                            out_file.write(f"{transcription[-1][0]} ({proc_time:.2f}s) | {text}\n")
                        elif args.timestamp:
                            out_file.write(f"{transcription[-1][0]} | {text}\n")
                        else:
                            out_file.write(f"{text}\n")
                        out_file.flush()

                    # Display current transcription in the terminal
                    if args.fastwhisper:
                        # Clear screen and display full transcription (Faster Whisper style)
                        os.system('cls' if os.name == 'nt' else 'clear')
                        for line in transcription:
                            if args.timestamp:
                                ts, txt, pt = line
                                if args.procdur:
                                    print(f"{ts} ({pt:.2f}s) | {txt}")
                                else:
                                    print(f"{ts} | {txt}")
                            else:
                                print(line)
                        print('', flush=True)
                    else:
                        # Use in-place update for continuous output (original Whisper style)
                        print("\r" + display_text, end="", flush=True)
                else:
                    # No audio data to process, sleep briefly to avoid CPU spinning
                    sleep(0.02)
            except Exception as e:
                print(f"\nError in transcription thread: {e}")
                break
    
    # ========== THREAD MANAGEMENT ==========
    # Start transcription in a separate thread so it doesn't block the main thread
    transcribe_thread = threading.Thread(target=transcribe_loop)
    transcribe_thread.daemon = True  # Thread will terminate when the main program exits
    transcribe_thread.start()
    
    # Show which options are being used for better traceability
    model_type = "Faster Whisper" if args.fastwhisper else "Original Whisper"
    print(f"Using {model_type} on {device} with {args.threads} threads")
    if args.fastwhisper:
        print(f"Optimizations: compute_type={compute_type}, beam_size=5, VAD filtering enabled")
    
    # Main thread just waits for keyboard interrupt (Ctrl+C)
    try:
        while transcribe_thread.is_alive():
            sleep(0.01)  # Reduced sleep time for faster responsiveness
    except KeyboardInterrupt:
        # Signal the transcription thread to stop
        running = False
        # Wait for transcription thread to finish (with timeout)
        transcribe_thread.join(timeout=1.0)

    # ========== FINAL OUTPUT ==========
    # Print the complete transcription at the end
    print("\n\nTranscription:")
    for line in transcription:
        if args.timestamp:
            if args.procdur:
                timestamp, text, proc_time = line
                print(f"{timestamp} ({proc_time:.2f}s) | {text}")
            else:
                timestamp, text, _ = line
                print(f"{timestamp} | {text}")
        else:
            print(line)
            
    # Clean up resources
    if out_file:
        out_file.close()


# ========== SCRIPT ENTRY POINT ==========
if __name__ == "__main__":
    main()