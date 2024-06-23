import argparse
import os
import numpy as np
import speech_recognition as sr
import torch
from datetime import datetime, timedelta
import datetime as dt
from queue import Queue
from time import sleep
from sys import platform
from threading import Thread
from faster_whisper import WhisperModel

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distil-medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", 
                                 "distil-medium", "distil-large", "distil-large-v2", "distil-large-v3"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    elif 'darwin' in platform:
        parser.add_argument("--default_microphone", default='Built-in',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            source = None
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
            if source is None:
                source = sr.Microphone(sample_rate=16000)
    else:
        source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    
    compute_type = 'int8' if 'distil' in model else 'auto'
    audio_model = WhisperModel(model, 
                               device='cuda' if torch.cuda.is_available() else 'cpu', 
                               compute_type=compute_type,
                               cpu_threads=8)


    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    print("Model loaded.\n")

    def transcribe_audio():
        nonlocal phrase_time
        while True:
            now = datetime.now(dt.UTC)
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                segments, info = audio_model.transcribe(audio_np, beam_size=12)
                text = ''.join(segment.text for segment in segments).strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line, end='')
                print('', end='\n', flush=False) # changed to False -- had potential flush bug where same expression flushed with each new message.
            else:
                sleep(0.02)

    transcribe_thread = Thread(target=transcribe_audio)
    transcribe_thread.start()

    try:
        while True:
            sleep(0.02)
    except KeyboardInterrupt:
        transcribe_thread.join()

    print("\n\nTranscription:")
    for line in transcription:
        print(line)

if __name__ == "__main__":
    main()

