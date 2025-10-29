#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-06-21T10:59:55-04:00

@author: nate
"""
import atexit
import contextlib
import datetime
import functools
import io
import json
import os
import re
import statistics
import subprocess as sp
import sys
import tempfile
import warnings
from collections.abc import Iterable
from itertools import groupby, zip_longest
from typing import Callable
from urllib.parse import urlparse

import bs4
import crawlpdfnames as cpn
import cv2
import ebook_to_audio as e2a
import ebooklib
import numpy as np
import pandas as pd
import popplerqt5
import PyQt5
import readabilipy as rp
import readtime
import tqdm
from ebook_to_audio import types
from ebook_to_audio.html_split import html_split
from ebook_to_audio.types import RunArgs
from loguru import logger
from PIL import Image
from PyQt5.QtCore import QBuffer
from PyQt5.QtGui import QImage


def _get_page_text(page):
    p0 = PyQt5.QtCore.QPointF(0,0)
    sizef = page.pageSizeF()
    r0 = PyQt5.QtCore.QRectF(p0, sizef)
    return page.text(r0)

def get_text_poppler(args: RunArgs):
    doc = popplerqt5.Poppler.Document.load(str(args.infile))
    text = ""
    for i in range(0, doc.numPages()):
        page = doc.page(i)
        breakpoint()
        page_text = _get_page_text(page)
        text = text+page_text + "\n" + e2a.types.page_char + "\n"
    text = e2a.preprocess.preprocess_text_lite(text, args.unspace)
    return text


def get_vlines_cv2(img, min_height_ratio=0.9):
    # Threshold to binary
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect lines with morphological operations
    height = binary.shape[0]
    width = binary.shape[1]
    #vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(height * 0.9)))
    #vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    return horizontal_lines


def show_mask(base_pil_img, overlay_mask):
    '''
    input:
    base_pil_img, a PIL Image to overlay on
    overlay_mask, a numpy grayscale array
    '''
    base_np = np.array(base_pil_img)
    red_overlay = np.zeros_like(base_np)
    red_overlay[..., 0] = 255
    output_np = np.where(overlay_mask[..., None] > 0, red_overlay, base_np)
    output_img = Image.fromarray(output_np)
    output_img.show()


def top_k_indices_with_min(arr, top_k, min_val):
    # Filter values above the threshold
    valid_indices = np.where(arr > min_val)[0]
    if len(valid_indices) == 0:
        return np.array([], dtype=int)  # No elements pass the filter
    # Get values that meet the condition
    valid_values = arr[valid_indices]
    # Get top-k indices among valid ones
    top_k = min(top_k, len(valid_values))  # adjust k if not enough
    top_k_local = np.argsort(-valid_values)[:top_k]
    # Map back to original indices
    return valid_indices[top_k_local]


def collapse_consecutive_blocks_to_medians(arr):
    if not arr:
        return []
    result = []
    start = 0
    n = len(arr)
    for i in range(1, n + 1):
        # End current block if next is not consecutive
        if i == n or arr[i][0] != arr[i-1][0] + 1:
            block = arr[start:i]
            values = [item[0] for item in block]
            median = int(statistics.median(values))
            ##########################################################
            # When we have mixed types in a block, determine the type
            # of the collapsed block via a hard-coded precedence
            # order.
            ##########################################################
            block_type = ""
            types = [item[1] for item in block]
            toc_items = filter(lambda item: isinstance(item, cpn.types.TOCItem), types)
            toc_items = list(toc_items)
            all_same_type =  all(map(lambda t0: t0 == types[0], types))
            if len(toc_items) == 1:
                block_type = toc_items[0]
            elif len(toc_items) > 1:
                logger.warning("Multiple TOCItems point to same location.")
                block_type = toc_items[0]
            elif all_same_type:
                block_type = types[0]
            else:
                logger.warning("Multiple items pointing to the same position.")
                breakpoint()
                block_type = types[0]
            result.append((median, block_type))
            start = i
    return result


def get_toc_dict(args: RunArgs):
    toc_items = {}
    curr_page = 0
    for level, toc_item in cpn.toc.get_toc_poppler(str(args.infile)):
        space = "  "*level
        page_no = toc_item.ld.pageNumber()
        if not page_no in toc_items:
            toc_items[page_no] = [toc_item]
        else:
            toc_items[page_no].append(toc_item)
    return toc_items

def get_text_poppler_line_split(args: RunArgs):

    DEBUG_MODE = False
    XPERCENT = 0.1
    top_k = 5
    ##################################################################
    doc = popplerqt5.Poppler.Document.load(str(args.infile))
    text = ""
    toc_dict = get_toc_dict(args)




    # ncols=0 for greatest chance that it doesn't look like trash.
    pbar = tqdm.tqdm(total=doc.numPages(), ncols=0, mininterval=1)

    for i in range(0, doc.numPages()):
        if pbar.update():
            # Print a newline on update
            print("")
        # Your Poppler-based PDF reading code here
        page = doc.page(i)
        img = page.renderToImage()
        buff = QBuffer()
        buff.open(QBuffer.ReadWrite)
        img.save(buff, "PNG")
        pil_img = Image.open(io.BytesIO(buff.data()))
        gray_img = pil_img.convert("L")
        vlines = get_vlines_cv2(np.array(gray_img))
        ##############################################################
        # Quantize vlines to {0,1}
        bm = np.where(vlines > 0, 1, 0)
        # Take row sums
        sums = sum(bm.T)
        # - Remove sums corresponding to rows that are less than XPERCENT red
        # - Get indices of k most red elements (unsorted)
        sums = sums / bm.shape[1]
        indices = top_k_indices_with_min(sums, top_k, XPERCENT).tolist()
        if not bm.shape[0] in indices:
            indices = indices + [bm.shape[0]-1]

        indices = [(item, 'hline') for item in indices]

        ##############################################################
        # Incorporate the splits from toc_dict
        ##############################################################
        new_indices = []
        for split in toc_dict.get(i+1, []):
            top = split.ld.top()
            if top > 1:
                top = 1

            s0 = int(top * bm.shape[0])
            logger.info(f"{i}: {split.title} [{s0}]")
            new = (s0, split)

            #indices.append( (s0, 'toc_item'))
            #new_indices.append( (s0, split.title + f" {split.ld.toString()}"))
            new_indices.append(new)

        indices = indices + new_indices

        ##############################################################
        
        indices = sorted(indices, key=lambda item: item[0])
        indices = collapse_consecutive_blocks_to_medians(indices)
        page_text = ''
        sizef = page.pageSizeF()
        p0 = PyQt5.QtCore.QPointF(0,0)

        if i >= 232:
            #breakpoint()
            pass

        for ind, ind_type in indices:
            p1 = PyQt5.QtCore.QPointF(sizef.width(), ind)
            r0 = PyQt5.QtCore.QRectF(p0, p1)

            ##########################################################
            if isinstance(ind_type, cpn.types.TOCItem):
                d0 = e2a.types.toc_item_char
            elif ind_type == 'hline':
                d0 = e2a.types.hline_char
            else:
                breakpoint()
            ##########################################################
            t0 = page.text(r0)
            #logger.info(f"{d0}, {ind}:")
            #print(t0)
            page_text = page_text + t0 + '\n' + d0 + "\n"
            p0 = PyQt5.QtCore.QPointF(0, p1.y())

        if i >= 232:
            #breakpoint()
            pass

        # Set those indices to solid lines
        bm2 = np.zeros_like(bm)
        for item in indices:
            bm2[item[0]-1, :] = 255

        #show_mask(pil_img, vlines)
        #show_mask(pil_img, bm2)
        ##############################################################
        #page_text = re.sub(f'({d0}[\n]+)+','\n'+d0+'\n', page_text)
        text = text+ page_text + "\n" + e2a.types.page_char + "\n"

    pbar.close()
    text = re.sub(f'[{d0}\n]+{e2a.types.page_char}', e2a.types.page_char, text)
    text = e2a.preprocess.preprocess_text_lite(text, args.unspace)
    return text



def get_text_poppler2(args: RunArgs):
    from poppler import PageRenderer, load_from_file
    pdf_document = load_from_file(args.infile)

    text = ""
    for i in range(0, pdf_document.pages):
        page = pdf_document.create_page(i)
        page_text = page.text()
        text = text+page_text.strip()+"\n"+e2a.types.page_char+"\n"

    return text
