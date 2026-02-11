"""
Speech training generator
Philipp D.
Feb. 2026
"""

import sys
import struct
import wave
import httpx
from time import sleep
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client()


def slice_text(text: str, tokens_per_slice: int) -> list[str]:
    output = []
    output_line = ""
    tokens = 0
    split_text = text.split("\n")
    for line in split_text:
        line_tokens = client.models.count_tokens(
            model="gemini-2.5-flash-preview-tts", contents=line
        ).total_tokens
        assert line_tokens is not None
        tokens += line_tokens
        if tokens >= tokens_per_slice:
            output.append(output_line)
            output_line = ""
            tokens = 0
        output_line += line + "\n"
    output.append(output_line)
    return output


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def generate_silence(duration_seconds, channels=1, rate=24000, sample_width=2):
    """Generate silence as PCM data."""
    num_samples = int(duration_seconds * rate)
    silence = struct.pack(
        "<" + "h" * num_samples * channels, *([0] * num_samples * channels)
    )
    return silence


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def parse_script_line(line: str) -> tuple[str, str] | None:
    """Parse a script line in 'Actor: Text' format. Returns (actor, text) or None."""
    if ":" not in line:
        return None
    parts = line.split(":", 1)
    actor = parts[0].strip()
    text = parts[1].strip()
    return (actor, text)


def filter_script_for_user(script_text: str, user_part: str) -> str:
    """Remove user's lines from script, leaving placeholders with pause info."""
    lines = script_text.strip().split("\n")
    filtered_lines = []

    for line in lines:
        if not line.strip():
            filtered_lines.append("")
            continue

        parsed = parse_script_line(line)
        if parsed is None:
            filtered_lines.append(line)
            continue

        actor, text = parsed
        if actor.lower() == user_part.lower():
            # Add a placeholder that will be replaced with silence
            word_count = count_words(text)
            pause_duration = word_count * 0.5
            filtered_lines.append(f"[PAUSE:{pause_duration}s]")
        else:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def read_pdf(client, doc_data, prompt):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            types.Part.from_bytes(
                data=doc_data,
                mime_type="application/pdf",
            ),
            prompt,
        ],
    )

    return response.text


def merge_audio_segments(segments, channels=1, rate=24000, sample_width=2):
    """Merge multiple audio segments (PCM data) into one."""
    merged = b""
    for segment in segments:
        merged += segment
    return merged


def process_script_with_pauses(script_text: str, user_part: str, client):
    """
    Process script to generate audio segments, inserting silence for user's lines.
    Returns the processed script ready for TTS (with pause instructions).
    """
    lines = script_text.strip().split("\n")
    processed_lines = []
    pause_info = []  # Track where silences should be inserted

    for line in lines:
        if not line.strip():
            processed_lines.append("")
            continue

        parsed = parse_script_line(line)
        if parsed is None:
            processed_lines.append(line)
            continue

        actor, text = parsed
        if actor.lower() == user_part.lower():
            # Calculate pause duration based on word count
            word_count = count_words(text)
            pause_duration = word_count * 0.5
            # Replace with instruction for silence
            processed_lines.append(
                f"[PAUSE for {word_count} words - {pause_duration} seconds]"
            )
            pause_info.append((len(processed_lines) - 1, pause_duration))
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines), pause_info


def create_audio_segment(client, text):
    """Generate audio for a text segment without any user filtering."""
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Orus",
                    )
                )
            ),
        ),
    )
    data = response.candidates[0].content.parts[0].inline_data.data
    return data


def create_audio_with_pauses(client, script_text, user_part=None):
    """Generate audio, inserting actual silence for user's lines."""
    prompt = "Read the following theater script aloud, including titles. Do not generate any text, only respond with audio.\n{}"
    if not user_part:
        # No user part specified, generate audio for entire script
        return create_audio_segment(client, prompt.format(script_text.strip()))

    # Split script into segments based on speaker
    lines = script_text.strip().split("\n")
    audio_segments = []
    current_segment = []

    for line in lines:
        parsed = parse_script_line(line)

        if parsed and parsed[0].lower() == user_part.lower():
            # This is a user line - generate audio for accumulated lines first
            if current_segment:
                segment_text = "\n".join(current_segment)
                audio_data = create_audio_segment(client, prompt.format(segment_text))
                audio_segments.append(audio_data)
                current_segment = []

            # Generate silence for user's line
            actor, text = parsed
            word_count = count_words(text)
            pause_duration = word_count * 0.5
            silence = generate_silence(pause_duration)
            audio_segments.append(silence)
        else:
            # Not a user line, accumulate it
            current_segment.append(line)

    # Don't forget the last segment
    if current_segment:
        segment_text = "\n".join(current_segment)
        audio_data = create_audio_segment(client, prompt.format(segment_text))
        audio_segments.append(audio_data)

    # Merge all segments
    return merge_audio_segments(audio_segments)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Please pass a PDF file URL, a file name, and an actor name.")
        sys.exit()

    doc_url = sys.argv[1]
    file_name = sys.argv[2]
    part = sys.argv[3]

    doc_data = httpx.get(doc_url).content

    prompt = "The document contains a script for a theater play. Extract the speech parts in 'Actor: Text' format, including titles. Respond with only the extracted text."
    print("Reading PDF...")
    pdf_text = read_pdf(client, doc_data, prompt)
    assert pdf_text is not None

    print("Preprocessing text...")
    sliced_pdf_text = slice_text(pdf_text, 7000)
    audio_segments = []
    for i, text_slice in enumerate(sliced_pdf_text):
        print(f"Generating audio {i+1}/{len(sliced_pdf_text)}...")
        data = create_audio_with_pauses(client, text_slice, user_part=part)
        audio_segments.append(data)
        if i < len(sliced_pdf_text) - 1:
            print("Waiting 1 minute as required by the API...")
            sleep(60)

    print("Merging audio segments...")
    merged_audio = merge_audio_segments(audio_segments)
    wave_file(file_name, merged_audio)
    print(f"Output saved at {file_name}.")
