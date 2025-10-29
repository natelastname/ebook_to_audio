#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-10-28T16:35:30-04:00

@author: nate
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Tuple

# --- simple regex filters ------------------------------------------------------

HEX_RE   = re.compile(r"^[0-9a-f]{16,64}$", re.IGNORECASE)
ISBN_RE  = re.compile(r"^(97[89][0-9]{10}|[0-9]{9}[0-9Xx])$")
YEAR_RE  = re.compile(r"^(1[5-9][0-9]{2}|20[0-4][0-9]|2050)$")
BRACKETS = re.compile(r"\s*[\(\[\{].*?[\)\]\}]\s*")

BAD_SEGMENTS = {
    "anna's archive", "annas archive", "anna’s archive",
}


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _ascii_fold(s: str) -> str:
    """Remove diacritics and normalize to plain ASCII."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _drop_junk_segments(segments: list[str]) -> list[str]:
    """Filter out obvious junk segments."""
    keep: list[str] = []
    for seg in segments:
        s = _nfkc(seg).strip()
        s = BRACKETS.sub(" ", s)
        s = " ".join(s.split())
        s_lower = s.lower()
        if not s:
            continue
        if s_lower in BAD_SEGMENTS:
            continue
        if HEX_RE.match(s.replace("-", "").replace(" ", "")):
            continue
        if ISBN_RE.match(s.replace("-", "").replace(" ", "")):
            continue
        if YEAR_RE.match(s):
            continue
        keep.append(s)
    return keep


def _safe_basename_from_filename(name: str, *, max_len: int = 64) -> str:
    """
    Clean up filename, remove noise, normalize Unicode,
    and return a Title-Cased version of the remaining text.
    """
    stem = Path(name).stem  # drop final extension
    candidate = _nfkc(stem)
    candidate = _ascii_fold(candidate)
    candidate = BRACKETS.sub(" ", candidate)
    candidate = " ".join(candidate.split())

    # Split on common separators just to remove junk segments, not reorder
    parts = re.split(r"[-–—,_]+", candidate)
    parts = _drop_junk_segments(parts)
    cleaned = " ".join(parts) if parts else candidate

    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "Untitled"

    # Title-case, limit length
    title_case = cleaned.title()

    return title_case


def _dedupe_path(path: Path) -> Path:
    """Append -2, -3, ... until free (directory semantics)."""
    if not path.exists():
        return path
    i = 2
    while True:
        p = path.with_name(f"{path.name}-{i}")
        if not p.exists():
            return p
        i += 1


def gen_outpath(outpath_parent: str | os.PathLike, infile: str | os.PathLike) -> Tuple[str, str]:
    """
    Create an output directory under `outpath_parent` based on a cleaned,
    title-cased version of the infile name. Example:

        gen_outpath('/home/Alice/tts_output',
                    'The Survival of the Wisest -- Jonas Edward Salk.pdf')
        -> ('The Survival Of The Wisest Jonas Edward Salk',
            '/home/Alice/tts_output/The Survival Of The Wisest Jonas Edward Salk')
    """
    parent = Path(outpath_parent)
    parent.mkdir(parents=True, exist_ok=True)

    src_name = Path(infile).name
    album = _safe_basename_from_filename(src_name)
    album = album.replace(' ', '')

    max_len = 64
    if len(album) > max_len:
        title_case = album[:max_len].rstrip()

    outdir = parent / album
    outdir = _dedupe_path(outdir)
    return album, str(outdir)
