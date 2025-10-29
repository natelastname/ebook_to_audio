#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-08T10:28:47-05:00

@author: nate
"""
from __future__ import unicode_literals

import atexit
import datetime
import datetime as dt
import functools
import json
import os
import re
import subprocess as sp
import sys
import tempfile
import wave
from collections.abc import Iterable
from itertools import groupby, zip_longest
from pathlib import Path

import ebook_to_audio as e2a
import ebooklib
import mutagen.mp3
import numpy as np
import pymupdf
import readtime
from ebook_to_audio import types
from ebook_to_audio.html_split import html_split
from ebook_to_audio.types import RunArgs, RunMode, SplitArgs, TOCStrat
from ebooklib import epub as ebooklib_epub
from loguru import logger
from mutagen.id3 import ID3, SYLT, Encoding
from PIL import Image
from piper.voice import PiperConfig, PiperVoice
from platformdirs import user_cache_dir

logger.remove()
logger.add(sys.stdout, level="INFO")

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

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

def subproc(cmd):
    result = sp.call(cmd.strip(), shell=True)
    return result

def gen_outpath(outpath_parent, infile):
    '''
    Create a directory under the directory `outpath_parent` based on the
    basename of infile. For example,
    `gen_outpath('/home/Alice/tts_output', './Track 1 - Album.mp3')`
    Creates the directory:
    `/home/Alice/tts_output/Track1Album/`
    '''
    (bname, ext) = os.path.splitext(os.path.basename(str(infile)))
    basename = e2a.preprocess.gen_basename(bname)
    #basename = e2a.cli.gen_basename(infile)
    res = os.path.join(outpath_parent, basename)
    os.makedirs(res, exist_ok=True)
    return basename, res

def mix_images(images):
    sizes = [img.size for img in images]
    if len(set(sizes)) > 1:
        raise ValueError("All images must have the same dimensions")
    arrays = [np.array(img, dtype=np.float32) for img in images]
    # Compute the average across all images
    mixed_array = np.mean(arrays, axis=0)
    # Convert back to uint8 and create PIL image
    mixed_image = Image.fromarray(mixed_array.astype(np.uint8))
    return mixed_image

def make_temp_dir():
    tempdir = tempfile.mkdtemp(prefix="e2a.", dir='/tmp')
    cmd = f"""
    rm -rf '{tempdir}'
    """
    delete_func = functools.partial(subproc, cmd)
    atexit.register(delete_func)
    return  tempdir

def pdf_test(infile: str):
    doc = pymupdf.open(infile)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def convert_to_text(infile, tempdir=None):
    if not tempdir:
        tempdir = make_temp_dir()
    outfile = os.path.join(tempdir, 'output.txt')
    outfile_raw = os.path.join(tempdir, 'output_raw.txt')
    infile_pdf = os.path.join(tempdir, 'input.pdf')
    ######################################################################
    # Conversion to text
    ######################################################################
    if not infile.endswith('.pdf'):
        cmd = f"""
        ebook-convert "{infile}" "{infile_pdf}" > /dev/null 2>&1
        """
        result = subproc(cmd)
        if result > 0:
            raise Exception('Failed to convert fle to PDF.')
        infile = infile_pdf
    text = pdf_test(infile)
    return text

######################################################################
# MP3 book keeping
######################################################################

def set_metadata_tag(vid_path, meta):
    outpath0 = os.path.dirname(vid_path)
    vid_name = os.path.basename(vid_path)
    album = meta['album']
    artist = meta['artist']
    desc = meta['description']
    track = meta['track']
    title = meta['title']
    cmd = f"""
    id3v2 --artist '{artist}' --track '{track}' --album '{album}' --comment '{desc}' --song '{title}' '{vid_path}'
    """
    subproc(cmd)
    return

def get_piper_voice(model_name="en_GB-alan-medium.onnx"):
    logger.debug('Loading voice...')

    model_file_path = Path(user_cache_dir("ebook_to_audio"))
    model_file_path = model_file_path / "models" / model_name
    voice = PiperVoice.load(model_file_path)
    return voice

def perform_tts(text, outfile, tempdir, voice):
    outfile_text = os.path.join(tempdir, "outfile_raw.txt")
    with open(outfile_text, "w+") as fp:
        fp.write(text)
    outfile_wav_raw = os.path.join(tempdir, 'outfile_raw.aac')
    wav_file = wave.open(outfile_wav_raw, 'w')
    logger.debug('synthesizing...')
    audio = voice.synthesize(text, wav_file, sentence_silence=0.1)
    logger.debug('Running ffmpeg...')
    cmd = f"""
    ffmpeg -hide_banner -loglevel error -y -i "{outfile_wav_raw}" -ab 64k "{outfile}"
    """
    subproc(cmd)
    return outfile

def _get_chunk_by_time(text0, seconds):
    '''
    Returns:
    - `chunk`: A leading chunk of the text that takes approximately
      `seconds` to read.
    - `rest`: The trailing remainder of the text.
    '''
    interval = datetime.timedelta(seconds=seconds)
    delta = datetime.timedelta(minutes=0)
    pattern = "\n"
    for m0 in re.finditer('\n', text0):
        start, end = m0.span()
        chunk = text0[0:start]
        rest = text0[end:]
        time = readtime.of_text(chunk).delta
        if time >= interval:
            return chunk, rest
    return text0, ''

def flatten_iter(items, depth: int = 0):
    """Given an iterable of nested iterables, yield non-iterables via DFS."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten_iter(x, depth=depth+1):
                yield sub_x
        else:
            #yield (depth, x)
            yield x

def flatten_iter_full(items, parents=[], depth: int = 0):
    """Given an iterable of nested iterables, yield non-iterables via DFS."""
    for x in items:
        if isinstance(x, tuple):
            assert len(x) == 2 and isinstance(x[0], ebooklib_epub.Section)
            parents_x = parents + [x[0]]
            for item0 in flatten_iter_full(x[1], parents=parents_x):
                yield item0
        elif isinstance(x, ebooklib_epub.Link):
            yield parents + [x]
        else:
            breakpoint()

class TOCMethods:
    @staticmethod
    def by_title(args: RunArgs, book):
        for items in flatten_iter_full(book.toc):
            title = '/'.join([i0.title for i0 in items])
            last = items[-1]
            href = last.href.split('#')[0]
            doc = book.get_item_with_href(href)
            if not doc:
                breakpoint()

            yield title, doc.content

    @staticmethod
    def by_fname(args: RunArgs, book):
        items = []
        for i, item in enumerate(flatten_iter(book.toc)):
            href = item.href
            doc_path = ""
            doc_id = ""
            m0 = re.match('(.*)#(.*)', item.href)
            if m0:
                doc_path, doc_id = m0.groups()
            else:
                doc_path = item.href

            # Add an index so that we preserve the order after sorting it by path
            items.append({
                'index': i,
                'path': doc_path,
                'id': doc_id,
            })
        # Sort by path so that groupby will work properly
        items = sorted(items, key=lambda item: item['path'])
        for key, group in groupby(items, lambda x: x['path']):
            group = [g0 for g0 in group]
            by_id = {g0['id']: g0 for g0 in group}
            group_ids = [g0['id'] for g0 in group]
            html = book.get_item_with_href(key)
            items = []

            first = True
            html_str = html.content
            for hr, name, text in html_split(html_str, key, group_ids):
                # The stuff "in the front" (before the first element
                # referenced by ID) has a path without any fragment.
                m0 = re.match('(.*)#(.*)', name)
                if not m0:
                    if not first:
                        breakpoint()
                    first = False
                    items.append((hr, text, 0))
                    continue
                href_base, href_id = m0.groups()
                if not href_id == "init" and not href_id in by_id:
                    breakpoint()

                if href_id == 'init':
                    index = 0
                else:
                    index = by_id[href_id]['index'] + 1
                items.append((hr, text, index))
            # Re-sort by index so that the original order is preserved
            '''
            for name, text,index in items:
                print(f"{name} - {index}")
            items = sorted(items, key=lambda i0: i0[2], reverse=True)
            '''

            for name, text, index in items:
                yield name, text

        return


    @staticmethod
    def html_items(args: RunArgs) -> e2a.types.TTSTrack:
        opts = {
            "ignore_ncx": True
        }
        book = ebooklib_epub.read_epub(args.infile, options=opts)
        tracks = []

        def should_skip(item):
            if isinstance(item, ebooklib_epub.EpubHtml):
                return False
            elif isinstance(item, ebooklib_epub.EpubItem):
                t0 = item.get_type()
                if t0 == ebooklib.ITEM_UNKNOWN:
                    # .html ends up here
                    return False
                elif t0 == ebooklib.ITEM_DOCUMENT:
                    return False
                else:
                    # It is something weird like an image or audio track
                    return True

        items = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        html_items = {item.get_id(): item for item in items}
        # Get the items in reading order
        reading_order = []
        for idref, _ in book.spine:
            if idref in html_items:
                reading_order.append(html_items[idref])


        for i0 in reading_order:
            skip = should_skip(i0)
            if skip:
                print(f'{i0} (skipped)')
                continue

            ordered = ebooklib_epub.get_pages_for_items([i0])
            print(f'{i0}')

            yield i0.id, i0.content


def get_toc_epub(args: RunArgs):
    opts = {
        "ignore_ncx": True
    }
    #book = ebooklib_epub.read_epub(args.infile, options=opts)
    book = ebooklib_epub.read_epub(args.infile)
    infile = args.infile
    title = book.title
    res = None

    if args.toc_strat == TOCStrat.DEFAULT:
        res = TOCMethods.html_items(args)
    elif args.toc_strat == TOCStrat.EPUB_FNAME:
        res = TOCMethods.by_fname(args, book)
    elif args.toc_strat == TOCStrat.EPUB_TITLE:
        res = TOCMethods.by_title(args, book)
    elif args.toc_strat == TOCStrat.EPUB_HTML_ITEMS:
        res = TOCMethods.html_items(args)
    else:
        raise NotImplementedError(args.toc_strat)

    for title, content in res:
        if args.run_mode == 'name':
            track = e2a.types.TTSTrack(title=title, text='')
            yield track
            continue
        text_content = content
        if not isinstance(content, str):
            text_content = e2a.util.html_str_to_text(content.decode())
        text_content = e2a.preprocess.preprocess_text(text_content, args.unspace)
        track = e2a.types.TTSTrack(title=title, text=text_content)
        yield track


def get_toc_pdf(args: RunArgs):
    text = e2a.get_pdf_text.get_text_poppler_line_split(args)
    text = e2a.preprocess.preprocess_text_lite(text, args.unspace)
    res = None
    if args.toc_strat == TOCStrat.DEFAULT:
        res = e2a.util.ChunkMethods.get_chunk_by_page(text)
    elif args.toc_strat == TOCStrat.PDF_CPN_TOC:
        res = e2a.util.ChunkMethods.cpn_pdf_toc(args, text)
    elif args.toc_strat == TOCStrat.PDF_PAGE:
        res = e2a.util.ChunkMethods.get_chunk_by_page(text)
    else:
        raise NotImplementedError(args.toc_strat)
    for track in res:
        if args.run_mode == 'name':
            track = e2a.types.TTSTrack(title=track.title, text='')
            yield track
            continue
        yield track

def get_toc(args: RunArgs):
    if str(args.infile).endswith("pdf"):
        return get_toc_pdf(args)
    return get_toc_epub(args)

def html_str_to_text(html: str):
    text_content = ""
    with tempfile.TemporaryDirectory() as tmpdir:
        tf = tempfile.mktemp(prefix=tmpdir, suffix=".html")
        with open(tf, 'w+') as fp:
            fp.write(html)

        text_content = e2a.util.convert_to_text(tf)
    return text_content

class ChunkMethods:

    @staticmethod
    def cpn_pdf_toc(args: RunArgs, text):
        import crawlpdfnames as cpn
        pages = text.split(e2a.types.page_char)
        curr_page = 0
        for level, toc_item in cpn.toc.get_toc_poppler(str(args.infile)):
            space = "  "*level
            page_no = toc_item.ld.pageNumber()
            left = f"{page_no:6}: {space}"
            print(left+toc_item.title)
            section = e2a.types.page_char.join(pages[curr_page:page_no])
            curr_page = page_no
            track = e2a.types.TTSTrack(text=section, title=toc_item.title)
            #breakpoint()
            yield track

    @staticmethod
    def identity_func(text):
        yield text

    @staticmethod
    def get_chunk_by_time(rest, seconds):
        while True:
            chunk, rest = _get_chunk_by_time(rest, seconds)
            yield chunk
            if rest.strip() == "":
                break
        return

    @staticmethod
    def get_chunk_by_page(text0):
        page_num = 0
        pages = re.split(e2a.types.page_char, text0)
        for i, page in enumerate(pages):
            page_num += 1
            track = e2a.types.TTSTrack(text=page, title=f"p. {page_num}")
            yield track
        return

######################################################################

def generate_with_sylt(
        text,
        outfile,
        tempdir
):
    logger.info("Running...")
    lyrics = []
    curr_time = 0
    out_sents = []
    lines = re.split(f'\n+', text.strip())
    outfile_temp, _ = os.path.splitext(outfile)

    outfile_temp_1 = outfile_temp + ".tmp1.wav"
    if os.path.isfile(outfile_temp_1):
        os.unlink(outfile_temp_1)

    outfile_temp_2 = outfile_temp + ".tmp2.wav"
    if os.path.isfile(outfile_temp_2):
        os.unlink(outfile_temp_2)

    voice = get_piper_voice()

    for line in lines:
        print(line)
        line0 = line
        with tempfile.TemporaryDirectory() as tempdir:

            outfile0 = os.path.join(tempdir, 'sent0.wav')
            e2a.util.perform_tts(line0, outfile0, tempdir, voice)
            if not os.path.isfile(outfile0):
                print("skipped")
                continue

            duration0 = get_wav_duration(outfile0)
            duration0 = int(duration0 * 1000)

            lyrics.append((line0, curr_time))
            #logger.info(f"[{round(curr_time,3):8}, {line0}")
            curr_time += duration0

            if os.path.isfile(outfile_temp_1):
                concat_audio_wave([outfile_temp_1, outfile0], outfile_temp_2)
                e2a.util.subproc(f"mv '{outfile_temp_2}' '{outfile_temp_1}'")
            else:
                concat_audio_wave([outfile0], outfile_temp_1)


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


def _convert_text(chunk_generator, outfile_mp3: str, tempdir: str, gen_lrc: bool=True):
    """Yield (track, final_mp3_path) after synthesis."""
    for track in chunk_generator:
        if not isinstance(track, e2a.types.TTSTrack):
            raise TypeError("chunk_generator yielded non-TTSTrack")

        print("###########################################################")
        print(track.text)
        print("###########################################################")

        # Strip control chars
        for ch in (e2a.types.sentence_char, e2a.types.pg_char, e2a.types.page_char):
            track.text = track.text.replace(ch, "")

        if gen_lrc:
            generate_with_sylt(track.text, outfile_mp3, tempdir)
        else:
            perform_tts(track.text, outfile_mp3, tempdir)

        yield track, outfile_mp3

def convert_text(
    chunk_gen,
    outpath: str,
    artist: str,
    tempdir: str,
    no_metadata: bool = False,
    gen_lrc: bool = True,
) -> None:
    outdir = Path(outpath)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = outdir / "raw.mp3"
    gen0 = _convert_text(chunk_gen, str(tmp_mp3), tempdir, gen_lrc=gen_lrc)

    for item_num, (track, _) in enumerate(gen0, start=1):
        z = f"{item_num:06d}"
        final_mp3 = outdir / f"{z}.mp3"
        meta = {
            "artist": artist,
            "title": f"{z} {track.title}",
            "album": artist,
            "track": item_num,
            "description": f"Created {dt.datetime.now().isoformat(timespec='minutes')}",
        }
        if not no_metadata:
            set_metadata_tag(str(tmp_mp3), meta)

        tmp_mp3.rename(final_mp3)
        if gen_lrc:
            sylt_to_lrc(meta['title'], meta['artist'], final_mp3)
