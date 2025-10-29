#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-12-08T10:04:25-05:00

@author: nate
"""
from __future__ import annotations

import atexit
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import argh
import ebook_to_audio as e2a
from ebook_to_audio.types import RunArgs, RunMode, SplitArgs, TOCStrat
from loguru import logger

# --------------------------- Small utilities ---------------------------

def run(cmd: list[str]) -> None:
    """Run a subprocess command with check=True, no shell."""
    logger.debug("Running: {}", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

def run_shell(cmd: str) -> int:
    """Last-resort shell runner (only when a string command is required)."""
    # Prefer list-style subprocess calls; this keeps shell usage explicit.
    logger.debug("Running (shell): {}", cmd)
    return subprocess.call(cmd, shell=True, executable="/bin/bash")

def strip_linearization_to_temp(input_pdf_path: Path) -> Path:
    """
    Use qpdf to strip linearization and write to a persistent temp file.
    File is removed on process exit.
    """
    input_pdf_path = input_pdf_path.resolve()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    def cleanup() -> None:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception as ex:
            logger.warning("Temp cleanup failed for {}: {}", tmp_path, ex)

    atexit.register(cleanup)
    run(["qpdf", str(input_pdf_path), "--qdf", "--normalize-content=y", str(tmp_path)])
    return tmp_path

def _coerce_suffix(p: Path) -> str:
    return p.suffix.lower().lstrip(".")

def _get_chunk_generator(args: RunArgs, infile: Path):
    """
    Return a chunk generator given the (possibly converted) infile.
    Mirrors your original logic but keeps it tight.
    """
    suffix = _coerce_suffix(infile)

    if suffix == "pdf":
        return e2a.util.get_toc(args)

    if suffix in {"azw3", "mobi", "djvu"}:
        # Convert to EPUB in a temp directory, then TOC
        with tempfile.TemporaryDirectory() as tdir:
            converted = (infile.stem + ".epub")
            converted = Path(tdir) / converted
            run(["ebook-convert", str(infile), str(converted)])
            args.infile = str(converted)
            return e2a.util.get_toc(args)

    if suffix == "epub":
        return e2a.util.get_toc(args)

    if suffix == "txt":
        text = infile.read_text(encoding="utf-8", errors="replace")
        if text.find(e2a.types.page_char) != -1:
            return e2a.util.ChunkMethods.get_chunk_by_page(text)
        return e2a.util.ChunkMethods.identity_func(text)

    raise ValueError(f"No chunk generator for '*.{suffix}' files.")

def _ensure_outdir(base_outpath: Optional[Path], infile: Path) -> tuple[str, Path]:
    """
    Compute album name and ensure the final outpath directory exists.
    """
    outpath = str(base_outpath or infile.parent)
    infile = str(infile)
    #album, outdir = e2a.util.gen_outpath(outpath, infile)
    album, outdir = e2a.outpath_generator.gen_outpath(outpath, infile)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    return album, outdir_path


# --------------------------- Core operations ---------------------------

def _write_toc_text(outfile: Path, toc: Iterable) -> None:
    """
    Stream the TOC pages to disk with page separators, avoiding big in-memory strings.
    """
    with outfile.open("w", encoding="utf-8") as fp:
        for i, toc_item in enumerate(toc):
            print("#####################################################")
            print(f"{i:4}: {toc_item.title}")
            print("#####################################################")
            page_txt = (toc_item.text or "").strip()
            fp.write(page_txt)
            fp.write(e2a.types.page_char + "\n\n")

def _maybe_strip_pdf(args: RunArgs) -> None:
    """If rm_linearization is set and file is PDF, replace args.infile with temp-stripped path."""
    p = Path(args.infile)
    if args.rm_linearization and _coerce_suffix(p) == "pdf":
        logger.info("Stripping PDF linearization…")
        stripped = strip_linearization_to_temp(p)
        args.infile = str(stripped)
        logger.info("Linearization stripped to temp: {}", stripped)

def split_txt(
    infile: str,
    outpath: Optional[str] = None,
    toc_strat: TOCStrat = TOCStrat.DEFAULT,
    unspace: bool = False,
    rm_linearization: bool = False,
) -> None:
    """
    Extract TOC text into a single .txt with page separators.

    Positional:
      infile: Path to PDF/EPUB/TXT/… file.

    Flags:
      --outpath PATH
      --toc-strat {auto,largest,flat,headings,...}  # whatever your TOCStrat supports
      --unspace / --no-unspace
      --rm-linearization
    """
    src = Path(infile).resolve()

    # Build a RunArgs that matches your library expectations.
    runargs = RunArgs(
        infile=str(src),
        outpath=str(Path(outpath) if outpath else src.parent),
        toc_strat=toc_strat,
        unspace=unspace,
        bilingual=False,
        run_mode="full",
        rm_linearization=rm_linearization,
    )

    _maybe_strip_pdf(runargs)

    # Where do we write?
    # Create a directory per source base name (like your original implementation).
    outdir = os.path.dirname(str(src))
    album, _ = e2a.outpath_generator.gen_outpath(outdir, src.stem)
    outfile = os.path.join(outdir, f"{album}.txt")

    logger.info("Writing text to {}", outfile)
    toc = e2a.util.get_toc(runargs)
    _write_toc_text(Path(outfile), toc)
    logger.success("Done: {}", outfile)



def tts(
    infile: str,
    *,
    outpath: Optional[str] = None,
    toc_strat: TOCStrat = TOCStrat.DEFAULT,
    unspace: bool = False,
    bilingual: bool = False,
    run_mode: RunMode = RunMode.NORMAL,
    rm_linearization: bool = False,
    gen_lrc: bool = True
) -> Optional[str]:
    """
    TTS pipeline. If --print-only != off, just prints TOC. Otherwise renders audio.

    Typical:
      run-tts book.pdf --outpath ./out --toc-strat auto --unspace --rm-linearization
    """
    e2a.get_model.auto_download()

    src = Path(infile).resolve()
    runargs = RunArgs(
        infile=str(src),
        outpath=str(Path(outpath) if outpath else src.parent),
        toc_strat=toc_strat,
        unspace=unspace,
        bilingual=bilingual,
        run_mode=run_mode,
        rm_linearization=rm_linearization,
        gen_lrc=gen_lrc
    )

    _maybe_strip_pdf(runargs)

    if runargs.run_mode != RunMode.NORMAL:
        toc = e2a.util.get_toc(runargs)
        for i, toc_item in enumerate(toc):
            if runargs.run_mode == "name":
                print(f"{i:4}: {toc_item}")
                continue
            if runargs.run_mode == "full":
                print("#####################################################")
                print(f"{i:4}: {toc_item.title}")
                print("#####################################################")
                print((toc_item.text or "").strip())
                print("#####################################################")
        return None

    album, final_outdir = _ensure_outdir(Path(runargs.outpath), Path(runargs.infile))
    chunk_gen = _get_chunk_generator(runargs, Path(runargs.infile))

    with tempfile.TemporaryDirectory() as tempdir:
        if runargs.bilingual:
            e2a.bilingual.convert_text_bilingual(chunk_gen, str(final_outdir), album, tempdir)
        else:
            e2a.util.convert_text(chunk_gen, str(final_outdir), album, tempdir)

    logger.success("Audio written to {}", final_outdir)
    return str(final_outdir)


# ------------------------------- CLI -----------------------------------

def e2a_cli() -> argh.ArghParser:
    parser = argh.ArghParser()
    parser.add_commands([
        split_txt,
        tts,
    ])
    parser.dispatch()
