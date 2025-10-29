#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-03-04T20:37:24-05:00

@author: nate
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from pydantic import BaseModel

# ----------------------------- Constants ------------------------------

sentence_char = "∙"
pg_char = "¶"
page_char = "📄"
hline_char = "✂"
toc_item_char = "⁉️"

# ----------------------------- Enums ----------------------------------

class TOCStrat(Enum):
    # Other
    DEFAULT = "default"
    # EPUB
    EPUB_FNAME = "fname"
    EPUB_TITLE = "title"
    EPUB_HTML_ITEMS = "html_items"
    # PDF
    PDF_PAGE = "pdf_page"
    PDF_CPN_TOC = "cpn_pdf_toc"

    def __str__(self) -> str:
        return self.value

class RunMode(Enum):
    NORMAL = "NORMAL"
    DRY_RUN_TOC = "DRY_RUN_TOC"
    DRY_RUN_FULL = "DRY_RUN_FULL"

    def __str__(self) -> str:
        return self.value


# ----------------------------- Dataclasses ----------------------------

@dataclass
class SplitArgs:
    infile: Path
    outpath: Optional[Path] = None
    toc_strat: TOCStrat = TOCStrat.DEFAULT
    unspace: bool = False
    rm_linearization: bool = False

@dataclass
class RunArgs:
    infile: Path
    outpath: Optional[Path]
    toc_strat: TOCStrat = TOCStrat.DEFAULT
    unspace: bool = False
    bilingual: bool = False
    run_mode: RunMode = RunMode.NORMAL
    rm_linearization: bool = False
    gen_lrc: bool = True

# ----------------------------- Utils ----------------------------------

def elide_text(string: Optional[str], max_len: int) -> str:
    if not string:
        return ""
    # Collapse blank lines + surrounding non-words; map form-feed to nothing.
    trunc = re.sub(r"\n+\W*", "⮐", string.replace("\x0c", ""))
    if len(trunc) <= max_len:
        return trunc
    return trunc[: max_len - 1] + "…"

# ----------------------------- Models ---------------------------------

class TTSTrack(BaseModel):
    text: str = ""
    title: Optional[str] = None

    def __str__(self) -> str:
        trunc1 = elide_text(self.title, 48)
        text0 = re.sub(r"\n", f"{sentence_char}\n", self.text or "")
        s0 = []
        s0.append("#####################################################")
        s0.append(f"TITLE: '{trunc1}'")
        s0.append("#####################################################")
        s0.append(text0 + ("" if (not text0 or text0.endswith("\n")) else "\n"))
        s0.append("####################################################")
        return "\n".join(s0)

    def __repr__(self) -> str:
        return self.__str__()
