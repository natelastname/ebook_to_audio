#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-08T10:28:47-05:00

@author: nate
"""
from __future__ import unicode_literals

import atexit
import datetime
import functools
import json
import os
import re
import subprocess as sp
import sys
import tempfile
from itertools import groupby, zip_longest
from urllib.parse import urlparse

import PyQt5
from loguru import logger

try:
    # python 3.10+
    from collections.abc import Iterable
except ImportError:
    # python < 3.10
    from collections import Iterable

import warnings
from typing import Callable

import bs4
import ebook_to_audio as e2a
import ebooklib
import epub_toc
import pandas as pd
import popplerqt5
import readabilipy as rp
import readtime
from ebook_to_audio import types
from ebook_to_audio.html_split import html_split
from ebook_to_audio.types import RunArgs


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
    import numpy as np
    from PIL import Image
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


def split_pdf_page(pdf_path):
    import pdfplumber
    from PIL import Image
    from rich.progress import track

    with tempfile.TemporaryDirectory() as tempdir:
        with pdfplumber.open(pdf_path) as pdf:
            i = 0
            for page in track(pdf.pages):
                im = page.to_image(resolution=300)
                dest = f"{str(i).zfill(6)}.png"
                dest = os.path.join(tempdir, dest)
                im.save(dest=dest)
                if i > 20:
                    break
                i += 1

            logger.info('Loading images...')
            width = 0
            height = 0
            images = []

            i = 0
            #for image_path in track(os.scandir(tempdir)):
            for image_path in os.scandir(tempdir):
                image = Image.open(image_path).convert('RGBA')
                w0, h0 = image.size
                if w0 > width:
                    width = w0
                if h0 > height:
                    height = h0
                images.append(image)
                #logger.info(f'{i:5}/{len(pdf.pages):5}')

                i += 1
            logger.info('Merging images...')

            bg_color = (0, 0, 0, 0)
            merged_image = Image.new('RGBA', (width, height), bg_color )
            new_images = []
            for image in images:
                #alpha_channel = merged_image.split()[3].point(lambda p: alpha)
                #image.putalpha(alpha_channel)
                padded_img = Image.new('RGBA', (width, height), bg_color )
                padded_img.paste(image, (0, 0), image)
                new_images.append(padded_img)

            img0 = mix_images(new_images)

            breakpoint()


        for page in pdf.pages:
            # Define regions (adjust coordinates based on your PDF)
            top = 100
            bottom = 100
            header = page.crop((0, 0, page.width, 100))  # Top 100px
            body = page.crop((0, 100, page.width, page.height - 100))  # Middle
            footer = page.crop((0, page.height - 100, page.width, page.height))  # Bottom 100px

            logger.info('header:')
            logger.info('body:')
            logger.info('footer:')
            breakpoint()

def pdf_to_text(pdf_path, num_pages: int | None = None):
    import pdfminer.layout
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextBoxHorizontal
    out_text = ""
    pages = []
    for i, page_layout in enumerate(extract_pages(pdf_path)):
        page = ""
        if num_pages and i > num_pages:
            break
        breakon = False
        types = []
        elts = []
        boxes = []
        for i, element in enumerate(page_layout):
            if not isinstance(element, pdfminer.layout.LTTextBoxHorizontal):
                continue
            text = element.get_text()
            text = e2a.preprocess.pp_normalize_whitespace(text)
            lines = text.strip().split('\n')
            # Skip text that matches typical page number patterns
            #print(f'[{i:4}] {text}')

            boxes = boxes + [l0.strip() for l0 in lines]

            page += text
            elts.append(element)


        print(json.dumps(boxes, indent=2))
        pages.append(boxes)
        out_text += page + "\x0c"

    return out_text

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

    cmd = f"""
    pdftotext "{infile}" "{outfile_raw}"
    """
    result = subproc(cmd)
    if result > 0:
        raise Exception('Failed to convert PDF to text.')

    with open(outfile_raw, 'r') as fp:
        text = fp.read()

    #text = pdf_test(infile)


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



def perform_tts(text, outfile, tempdir, piper_exe_path, model_file_path):
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
    outfile_wav_raw = os.path.join(tempdir, 'outfile_raw.aac')
    cmd = f"""
    '{piper_exe_path}' --quiet --model '{model_file_path}' --output_file '{outfile_wav_raw}' < '{outfile_text}'
    """
    subproc(cmd)
    cmd = f"""
    ffmpeg -hide_banner -loglevel error -y -i "{outfile_wav_raw}" -ab 64k "{outfile}"
    """
    subproc(cmd)
    return outfile

######################################################################

######################################################################

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
            assert len(x) == 2 and isinstance(x[0], ebooklib.epub.Section)
            parents_x = parents + [x[0]]
            for item0 in flatten_iter_full(x[1], parents=parents_x):
                yield item0
        elif isinstance(x, ebooklib.epub.Link):
            yield parents + [x]
        else:
            breakpoint()

def check_coverage(found, book):
    for link in flatten_iter(book.toc):
        href = link.href
        href_base, href_id = re.match('(.*)#?(.*)', href).groups()

        if href not in found:
            # We did not find this
            print("Not encountered:")
            pass

        breakpoint()
        pass

class TOCMethods:
    @staticmethod
    def by_title(args: RunArgs, book):
        # TODO: Detect duplication

        counter = 0
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
        # TODO: Detect duplication
        contents = [elm.content for elm in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]

        items = []
        for i, item in enumerate(flatten_iter(book.toc)):
            href = item.href
            doc = book.get_item_with_href(href)
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
            for name, text in html_split(html_str, key, group_ids):
                # The stuff "in the front" (before the first element
                # referenced by ID) has a path without any fragment.
                m0 = re.match('(.*)#(.*)', name)
                if not m0:
                    if not first:
                        breakpoint()
                    first = False
                    items.append((name, text, 0))
                    continue
                href_base, href_id = m0.groups()
                if not href_id == "init" and not href_id in by_id:
                    breakpoint()

                if href_id == 'init':
                    index = 0
                else:
                    index = by_id[href_id]['index'] + 1
                items.append((name, text, index))

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
        book = ebooklib.epub.read_epub(args.infile, options=opts)
        tracks = []

        def should_skip(item):
            if isinstance(item, ebooklib.epub.EpubHtml):
                return False
            elif isinstance(item, ebooklib.epub.EpubItem):
                t0 = item.get_type()
                if t0 == ebooklib.ITEM_UNKNOWN:
                    # .html ends up here
                    return False
                elif t0 == ebooklib.ITEM_DOCUMENT:
                    return False
                else:
                    # It is something weird like an image or audio track
                    return True

        for i0 in book.items:
            skip = should_skip(i0)
            if skip:
                print(f'{i0} (skipped)')
                continue
            print(f'{i0}')

            yield i0.id, i0.content


def get_toc_epub(args: RunArgs):
    opts = {
        "ignore_ncx": True
    }
    #book = ebooklib.epub.read_epub(args.infile, options=opts)
    book = ebooklib.epub.read_epub(args.infile)
    infile = args.infile
    title = book.title
    res = None

    if args.toc_strat == "default":
        res = TOCMethods.html_items(args)
    elif args.toc_strat == "fname":
        res = TOCMethods.by_fname(args, book)
    elif args.toc_strat == "title":
        res = TOCMethods.by_title(args, book)
    elif args.toc_strat == "html_items":
        res = TOCMethods.html_items(args)
    else:
        raise NotImplementedError(strat)

    for title, content in res:
        if args.print_only == 'name':
            track = e2a.types.TTSTrack(title=title, text='')
            yield track
            continue
        text_content = content
        if not isinstance(content, str):
            text_content = e2a.util.html_str_to_text(content.decode())
        text_content = e2a.preprocess.preprocess_text(text_content, args.unspace)
        track = e2a.types.TTSTrack(title=title, text=text_content)
        yield track


def pdf_test(infile: str):
    '''
    infile = str(args.infile)
    text = e2a.util.convert_to_text(infile, tempdir=tempdir)
    text = e2a.preprocess.preprocess_text(text, args.unspace)
    chunk_generator = e2a.util.ChunkMethods.get_chunk_by_page(text)
    '''
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(infile)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    return text
    ##################################################################
    doc1 = pymupdf.open(args.infile)
    pagerange = list(range(0,25))
    data = pymupdf4llm.to_markdown(
        args.infile,
        pages=pagerange
    )
    splitted = re.split('\\s+\n#+ (.*)', data)
    for chap in splitted:
        chap = re.sub('\n\\-\\-\\-\\-\\-', '', chap)
        print(chap)
        breakpoint()

    #doc2 = pymupdf4llm.to_markdown(infile)
    #reader = pymupdf4llm.LlamaMarkdownReader()
    #data = reader.load_data(infile)

    #data = pymupdf4llm.to_markdown(args.infile, page_chunks=True)
    #toc_items = [page['toc_items'] for page in data]
    #toc_items = list(filter(lambda item: item, toc_items))

    for i, page in enumerate(data):
        toc = page['toc_items']
        print('#######################################################')
        print(f"{i:5}: {toc}")
        print('#######################################################')
        text0 = page['text']
        text1 = e2a.preprocess.preprocess_text(text, args.unspace)
        print(text1)
        print('#######################################################')
        #processed1 = e2a.preprocess.pp_remove_newlines(page['text'])
        breakpoint()
        pass

    return


def get_text_pdfplumber(args: RunArgs):
    import pdfplumber
    with pdfplumber.open(str(args.infile)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            im = page.to_image()
            logger.info(f'##### Page {i}')
            logger.info(f'Lines: {len(page.lines)}')
            im.draw_rects(page.extract_words())
            im.draw_lines(page.lines)



######################################################################
# Get text methods
######################################################################

def get_text_pdf(args: RunArgs):
    #with tempfile.TemporaryDirectory() as tempdir:

    ##############################################################
    # It is anticipated that multiple pdf2text methods will be
    # required for different situations.
    ##############################################################
    PDF_TO_TEXT_MODE = "Poppler"
    if PDF_TO_TEXT_MODE == "Poppler":
        text = e2a.get_pdf_text.get_text_poppler_line_split(args)
    else:
        breakpoint()

    text = e2a.preprocess.preprocess_text_lite(text, args.unspace)

    return text

def get_toc_pdf(args: RunArgs):
    text = get_text_pdf(args)
    res = None
    if args.toc_strat == "default":
        res = e2a.util.ChunkMethods.get_chunk_by_page(text)
    elif args.toc_strat == "cpn_pdf_toc":
        res = e2a.util.ChunkMethods.cpn_pdf_toc(args, text)
    elif args.toc_strat == "pdf_page":
        res = e2a.util.ChunkMethods.get_chunk_by_page(text)
    else:
        raise NotImplementedError(strat)

    for track in res:
        if args.print_only == 'name':
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
            breakpoint()
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

def _convert_text(
        chunk_generator,
        outfile,
        tempdir,
        piper_exe_path,
        model_file_path
):
    ''' Generates (Track, Outfile) pairs for speech synthesis.'''
    item_num = 0
    for track in chunk_generator:
        if not isinstance(track, e2a.types.TTSTrack):
            raise Exception("Chunk_generator yielded non-Track instance")

        print("###########################################################")
        print(track.text)
        print("###########################################################")

        track.text = re.sub(types.sentence_char, "", track.text)
        track.text = re.sub(types.pg_char, "", track.text)
        track.text = re.sub(types.page_char, "", track.text)

        perform_tts(
            track.text,
            outfile,
            tempdir,
            piper_exe_path,
            model_file_path
        )
        
        yield track, outfile
        item_num += 1

def convert_text(
        chunk_gen,
        outpath,
        artist,
        tempdir,
        piper_exe_path,
        model_file_path,
        no_metadata=False
):
    '''
    Perform text to speech synthesis.
    Supports chunking and Mp3 metadata.
    '''
    outfile = os.path.join(outpath, "raw.mp3")
    coro = _convert_text(
        chunk_gen,
        outfile,
        tempdir,
        piper_exe_path,
        model_file_path
    )
    i = 0
    for item_num, (track, tmpfile) in enumerate(coro):
        zfilled = f"{item_num+1}".zfill(6)
        final_outfile = f"{zfilled}.mp3"
        final_outfile = os.path.join(os.path.dirname(tmpfile), final_outfile)
        dt0 = datetime.datetime.now()
        desc = f"Created {dt0.isoformat(timespec='minutes')}"

        metadata = {
            'artist': artist,
            'title': f"{zfilled} {track.title}",
            'album': artist,
            'track': item_num + 1,
            'description': desc
        }
        if not no_metadata:
            set_metadata_tag(tmpfile, metadata)
        cmd = f"""
        mv '{outfile}' '{final_outfile}'
        """
        subproc(cmd)
        i = i + 1
    return
