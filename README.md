# Real Time Whisper Transcription

![Demo gif](demo.gif)

This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

## Installation

To install dependencies simply run:
```
pip install -r requirements.txt
```
in an environment of your choosing.

Whisper also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

## Usage

Run the transcription script with:

```
python transcribe.py
```

By default, this will use the original Whisper implementation with the "medium" model.

### Options

- `--fastwhisper`: Toggle to use the Faster Whisper implementation (faster and more memory efficient)
- `--model`: Select model to use (tiny, base, small, medium, large, etc.)
- `--energy_threshold`: Set energy level for microphone detection
- `--record_timeout`: Set how real-time the recording is (in seconds)
- `--phrase_timeout`: Set empty space between recordings before considering it a new line
- `--timestamp`: Add timestamps to transcription entries
- `--procdur`: Include processing duration after timestamp
- `--out`: Specify output file to write transcription results

### Examples

Use Faster Whisper with the distilled medium model:
```
python transcribe.py --fastwhisper --model distil-medium
```

Use original Whisper with timestamps and processing duration:
```
python transcribe.py --model small --timestamp --procdur
```

For more information on the original Whisper, see: https://github.com/openai/whisper  
For more information on Faster Whisper, see: https://github.com/guillaumekln/faster-whisper

## Benchmarking

To benchmark the transcription speed on a single audio file, use the `transcribe_benchmark.py` script:

```
python transcribe_benchmark.py --input /path/to/audio/file.mp3
```

### Benchmark Options

- `--model`: Select model to use (tiny, base, small, medium, large, distil-medium, etc.)
- `--non_english`: Don't use the English-specific model
- `--compute_type`: Set compute type for model inference (auto, int8, float16, etc.)
- `--cpu_threads`: Set number of CPU threads to use for processing
- `--verbose`: Print detailed information during processing

### Example

Benchmark with the distilled large model and verbose output:
```
python transcribe_benchmark.py --input sample.wav --model distil-large --verbose
```

The script will output:
- Transcription time
- Realtime factor (how much faster/slower than realtime)
- Processing speed
- The complete transcription
- Saves the transcription to a text file with the same name as the input file

The code in this repository is public domain.