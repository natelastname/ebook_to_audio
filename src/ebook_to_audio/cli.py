#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-08T10:04:25-05:00

@author: nate
"""

import argparse
import atexit
import functools
import os
import re
import subprocess
import sys
import tempfile

import bs4
import rich_click as click
from dataclass_click import (argument, dataclass_click, option,
                             register_type_inference)
from loguru import logger

import ebook_to_audio as e2a
from ebook_to_audio.types import RunArgs, SplitArgs, TOCStrat


def subproc(cmd):
    shlex.split(cmd)
    result = sp.call(cmd.strip(), shell=True, executable='/bin/bash')
    return result

def split_epub(epub_path: str, *, toc_strat: TOCStrat = "default",
               print_only: bool = False):
    outpath = os.path.dirname(epub_path)
    outdir = os.path.basename(epub_path).split('.')[:-1]
    outdir = '.'.join(outdir)
    os.makedirs(outdir, exist_ok=True)
    toc = e2a.util.get_toc(epub_path, toc_strat=toc_strat)
    i = 0
    for toc_item in toc:
        print(f"{i:4}: {toc_item}")
        title = toc_item.title.split('.')
        title = '.'.join(title[:-1])
        outfile = os.path.join(outdir, title+".txt")
        with open(outfile, "w+") as fp:
            fp.write(toc_item.text)
        #track = TTSTrack(text=text_content, title=toc_item)
        i += 1


def strip_linearization_to_temp(input_pdf_path: str) -> str:
    # Create a persistent temp file path (deleted on exit)
    temp_dir = tempfile.gettempdir()
    base_name = os.path.basename(input_pdf_path)
    temp_pdf_path = os.path.join(temp_dir, f"{base_name}")
    # Ensure it's deleted at program exit
    def cleanup():
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
    atexit.register(cleanup)
    # Run qpdf to strip linearization
    subprocess.run(["qpdf", input_pdf_path, "--qdf", "--normalize-content=y", temp_pdf_path], check=True)
    return temp_pdf_path


@click.command()
@dataclass_click(SplitArgs)
def split_txt(args: SplitArgs):

    if not args.outpath:
        path = args.infile
        outpath = os.path.dirname(path)
    else:
        path = args.infile
        outpath = args.outpath

    outfile = os.path.basename(path)
    outfile, _ = os.path.splitext(outfile)
    outfile = e2a.preprocess.gen_basename(outfile)

    # Create a directory for multiple output files (NOt sure if needed)
    outdir = os.path.join(outpath, outfile)
    os.makedirs(outdir, exist_ok=True)

    outfile = outfile + ".txt"
    outfile = os.path.join(outdir, outfile)
    logger.info(outfile)

    if args.rm_linearization:
        logger.info("Stripping linearization...")
        args.infile = strip_linearization_to_temp(args.infile)
        logger.info("Done.")


    runargs = e2a.types.RunArgs(
        infile=args.infile,
        outpath=outdir,
        toc_strat=args.toc_strat,
        unspace=args.unspace,
        bilingual=False,
        print_only='full',
        rm_linearization=args.rm_linearization
    )

    toc = e2a.util.get_toc(runargs)


    combined = ""
    for i, toc_item in enumerate(toc):
        print("#####################################################")
        print(f"{i:4}: {toc_item.title}")
        print("#####################################################")
        page_txt = toc_item.text.strip()
        print("#####################################################")
        combined += page_txt
        combined += e2a.types.page_char + '\n\n'

    with open(outfile, 'w+') as fp:
        fp.write(combined)
    return

######################################################################

def run0(args: RunArgs):
    """Perform TTS synthesis"""
    e2a.get_model.auto_download()
    album, outpath = e2a.util.gen_outpath(args.outpath, args.infile)
    os.makedirs(outpath, exist_ok=True)
    with tempfile.TemporaryDirectory() as tempdir:
        chunk_generator = None

        infile = str(args.infile)
        if infile.endswith('pdf'):
            chunk_generator = e2a.util.get_toc(args)

        if infile.endswith("azw3") or infile.endswith("mobi") or infile.endswith("djvu"):
            converted = os.path.basename(infile).split('.')[0:-1]
            converted = '.'.join(converted)
            converted = f"{converted}.epub"
            converted = os.path.join(tempdir, converted)
            cmd = f"ebook-convert '{infile}' '{converted}'"
            e2a.util.subproc(cmd)
            args.infile = converted
            # args.print_only
            chunk_generator = e2a.util.get_toc(args)

        if infile.endswith("epub"):
            chunk_generator = e2a.util.get_toc(args)

        if infile.endswith("txt"):
            text = open(infile, 'r').read()
            if text.find(e2a.types.page_char) != -1:
                chunk_generator = e2a.util.ChunkMethods.get_chunk_by_page(text)
            else:
                chunk_generator = e2a.util.ChunkMethods.identity_func(text)

        if not chunk_generator:
            infile = os.path.basename(infile)
            raise Exception(f"No chunk generator for '{infile}'.")


        if args.bilingual:
            e2a.bilingual.convert_text_bilingual(
                chunk_generator,
                outpath,
                album,
                tempdir
            )
            return outpath

        e2a.util.convert_text(
            chunk_generator,
            outpath,
            album,
            tempdir
        )
    return outpath


@click.command()
@dataclass_click(RunArgs)
def do_tts(args: RunArgs):
    """Compute and print table of contents"""
    if args.print_only == 'off':
        return run0(args)
    toc = e2a.util.get_toc(args)
    for i, toc_item in enumerate(toc):
        if args.print_only == 'name':
            print(f"{i:4}: {toc_item}")
            continue
        if args.print_only == 'full':
            print("#####################################################")
            print(f"{i:4}: {toc_item.title}")
            print("#####################################################")
            print(toc_item.text.strip())
            print("#####################################################")

######################################################################

@click.group()
def cli():
    pass

def e2a_cli():

    cli.add_command(do_tts)
    cli.add_command(split_txt)
    cli()
    #do_convert()
    #get_toc(args)
    pass
