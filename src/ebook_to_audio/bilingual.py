#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-04-14T20:17:26-04:00

@author: nate
"""
import datetime
import os
import re
import tempfile
import warnings
import wave

import argostranslate.package
import argostranslate.translate
import mutagen.mp3
from loguru import logger
from mutagen.id3 import ID3, SYLT, Encoding

import ebook_to_audio as e2a

# Why does this sentence not translate from english to spanish?
# ’He was educated Upper Canada College, Toronto, at Eton, at the University of Strasbourg and, after a spell of National Service in the Navy, at New College, Oxford, where he took a degree in French and Russian.’

def concat_audio_wave(audio_clip_paths, output_path):
    data = []
    for clip in audio_clip_paths:
        w = wave.open(clip, "rb")
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
    output = wave.open(output_path, "wb")
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()

def get_argo_package(from_code, to_code):
    available_packages = argostranslate.package.get_available_packages()
    for pkg in available_packages:
        if pkg.from_code == from_code and pkg.to_code == to_code:
            return pkg
    raise Exception("Could not find an argo package.")

def get_languages(from_code, to_code):
    pkg = get_argo_package(from_code, to_code)
    download_path = pkg.download()
    argostranslate.package.install_from_path(download_path)
    installed_languages = argostranslate.translate.get_installed_languages()
    from_lang = list(filter(lambda item: item.code == from_code, installed_languages))[0]
    to_lang = list(filter(lambda item: item.code == to_code, installed_languages))[0]
    return from_lang, to_lang

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration


def perform_tts_wav(text, outfile, tempdir, piper_exe_path, model_file_path):
    """

    Args:
        text: The text to convert.
        outfile: The output mp3.
        tempdir: A temporary directory to dump files used during
            conversion.
        piper_exe_path: The path to the piper TTS executable.
        model_file_path: The Piper model file

    Returns:
        outfile, the path to the resulting MP3.
    """

    outfile_text = os.path.join(tempdir, "outfile_raw.txt")
    with open(outfile_text, "w+") as fp:
        fp.write(text)
    outfile = os.path.join(tempdir, 'outfile_raw.aac')
    cmd = f"""
    '{piper_exe_path}' --quiet --model '{model_file_path}' --output_file '{outfile}' < '{outfile_text}'
    """
    subproc(cmd)
    return outfile

def format_timestamp(seconds: float) -> str:
    """Convert seconds (float) to LRC timestamp format [mm:ss.xx]."""
    minutes = int(seconds // 60)
    secs = seconds - (minutes*60)

    rem = int((secs - int(secs))*100)
    secs = int(secs)
    ts0 = f"[{minutes:02d}:{secs:02d}.{rem:02d}]"
    return ts0

def sylt_to_lrc(title: str, artist: str, mp3_file):
    lrc_file, _ = os.path.splitext(mp3_file)
    lrc_file = lrc_file + ".lrc"
    # Load MP3 and read ID3 tags
    audio = ID3(mp3_file)
    sylt_frames = audio.getall("SYLT")
    # Open LRC file for writing
    with open(lrc_file, 'w', encoding='utf-8') as f:
        # Optional: Add LRC metadata (title, artist, etc.)
        f.write(f"[ti: {title}]\n")
        f.write(f"[ar: {artist}]\n")
        f.write("\n")
        for sylt in sylt_frames:
            for text, timestamp in sylt.text:
                lrc_time = format_timestamp(timestamp/1000)
                line = f"{lrc_time} {text}\n"
                f.write(line)
    return


def perform_tts_bilingual(
    text,
    outfile,
    tempdir,
    translation
):
    from_code = "de"
    to_code = "en"
    model0 = '/home/nate/piper/models/en_GB-alan-medium.onnx'
    model1 = '/home/nate/piper/models/es_MX-claude-high.onnx'
    piper_exe_path = "/home/nate/piper/install/piper"
    if os.path.isfile(outfile):
        os.unlink(outfile)
    ##################################################################
    logger.info("Running...")
    lyrics = []
    curr_time = 0
    out_sents = []
    sentences = re.split(f'\n', text.strip())
    outfile_temp, _ = os.path.splitext(outfile)
    outfile_temp_1 = outfile_temp + ".tmp1.wav"
    if os.path.isfile(outfile_temp_1):
        os.unlink(outfile_temp_1)

    outfile_temp_2 = outfile_temp + ".tmp2.wav"
    if os.path.isfile(outfile_temp_2):
        os.unlink(outfile_temp_2)

    for sent in sentences:
        sent0 = sent
        sent1 = translation.translate(sent)
        with tempfile.TemporaryDirectory() as tempdir:

            outfile0 = os.path.join(tempdir, 'sent0.wav')
            e2a.util.perform_tts(
                sent0,
                outfile0,
                tempdir,
                piper_exe_path,
                model0
            )
            if not os.path.isfile(outfile0):
                continue

            duration0 = get_wav_duration(outfile0)
            duration0 = int(duration0 * 1000)

            lyrics.append((sent0, curr_time))
            logger.info(f"[{round(curr_time,3):8}, {from_code}]: {sent0}")
            curr_time += duration0

            outfile1 = os.path.join(tempdir, 'sent1.wav')
            e2a.util.perform_tts(
                sent1,
                outfile1,
                tempdir,
                piper_exe_path,
                model1
            )
            if not os.path.isfile(outfile1):
                continue

            duration1 = get_wav_duration(outfile1)
            duration1 = int(duration1 * 1000)

            lyrics.append((sent1, curr_time))
            logger.info(f"[{round(curr_time,3):8}, {to_code}]: {sent1}")
            curr_time += duration1
            # Concatenate files
            if os.path.isfile(outfile_temp_1):
                concat_audio_wave([outfile_temp_1, outfile0, outfile1], outfile_temp_2)
                e2a.util.subproc(f"mv '{outfile_temp_2}' '{outfile_temp_1}'")
            else:
                concat_audio_wave([outfile0, outfile1], outfile_temp_1)


    cmd = f"""
    ffmpeg -hide_banner -loglevel error -y -i "{outfile_temp_1}" -ab 64k "{outfile}"
    """
    e2a.util.subproc(cmd)
    if os.path.isfile(outfile_temp):
        os.unlink(outfile)

    m0 = mutagen.mp3.MP3(outfile)
    if m0.tags is None:
        m0.tags = mutagen.id3.ID3()

    # save ID3v2.3 only without ID3v1 (default is ID3v2.4)
    m0.save(v1=0, v2_version=3)

    audio = ID3(outfile)
    meta = SYLT(encoding=Encoding.UTF8, lang='eng', format=2, type=1, text=lyrics)
    audio.add(meta)
    audio.save(v2_version=3)

    return outfile


def _convert_text_bilingual(
    chunk_generator,
    outfile,
    tempdir
):
    ''' Generates (Track, Outfile) pairs for speech synthesis.'''
    item_num = 0

    from_code = "es"
    to_code = "en"
    from_lang, to_lang = get_languages(from_code, to_code)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        translation = from_lang.get_translation(to_lang)

    for track in chunk_generator:
        if not isinstance(track, e2a.types.TTSTrack):
            raise Exception("Chunk_generator yielded non-Track instance")

        print("###########################################################")
        print(track.text)
        print("###########################################################")

        track.text = re.sub(e2a.types.sentence_char, "", track.text)
        track.text = re.sub(e2a.types.pg_char, "", track.text)
        track.text = re.sub(e2a.types.page_char, "", track.text)

        perform_tts_bilingual(
            track.text,
            outfile,
            tempdir,
            translation
        )
        
        yield track, outfile
        item_num += 1


def convert_text_bilingual(
        chunk_gen,
        outpath,
        artist,
        tempdir
):
    '''
    Perform text to speech synthesis.
    Supports chunking and Mp3 metadata.
    '''
    outfile = os.path.join(outpath, "raw.mp3")
    coro = _convert_text_bilingual(
        chunk_gen,
        outfile,
        tempdir
    )
    i = 0
    for item_num, (track, tmpfile) in enumerate(coro):
        zfilled = f"{item_num+1}".zfill(6)
        final_outfile = f"{zfilled}.mp3"
        final_outfile = os.path.join(os.path.dirname(tmpfile), final_outfile)
        dt0 = datetime.datetime.now()
        desc = f"Created {dt0.isoformat(timespec='minutes')}"

        track_title = f"{zfilled} {track.title}"
        metadata = {
            'artist': artist,
            'title': track_title,
            'album': artist,
            'track': item_num + 1,
            'description': desc
        }
        e2a.util.set_metadata_tag(tmpfile, metadata)
        cmd = f"""
        mv '{outfile}' '{final_outfile}'
        """
        e2a.util.subproc(cmd)
        sylt_to_lrc(track_title, artist, final_outfile)
        i = i + 1
    return
