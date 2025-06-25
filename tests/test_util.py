#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-08T10:04:25-05:00

@author: nate
"""
import tempfile
import tomllib

import os

import ebook_to_audio as e2a

def test_pdf_1():
    basedir = os.path.dirname(__file__)
    infile = os.path.join(basedir, "test_data", '3pages.pdf')

    e2a_conf = os.path.expanduser("~/.ebook_to_audio.toml")
    with open(e2a_conf, 'rb') as fp:
        data = tomllib.load(fp)
    piper_exe_path = data['piper_exe_path']
    model_file_path = data['model_file_path']

    with tempfile.TemporaryDirectory() as tempdir:
        outpath = e2a.cli.do_convert(
            infile,
            tempdir,
            piper_exe_path,
            model_file_path
        )
        assert len(os.listdir(outpath)) == 3


def test_epub_1():
    basedir = os.path.dirname(__file__)
    infile = os.path.join(basedir, "test_data", '3chapters.epub')
    e2a_conf = os.path.expanduser("~/.ebook_to_audio.toml")
    with open(e2a_conf, 'rb') as fp:
        data = tomllib.load(fp)
    piper_exe_path = data['piper_exe_path']
    model_file_path = data['model_file_path']

    with tempfile.TemporaryDirectory() as tempdir:
        outpath = e2a.cli.do_convert(
            infile,
            tempdir,
            piper_exe_path,
            model_file_path
        )
        assert len(os.listdir(outpath)) == 3
