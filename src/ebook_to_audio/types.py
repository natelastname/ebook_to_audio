#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-03-04T20:37:24-05:00

@author: nate
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import click
from dataclass_click import argument, dataclass_click, option, register_type_inference
from pydantic import BaseModel

sentence_char = "∙"
pg_char = '¶'
page_char = "📄"
hline_char = '✂'
toc_item_char = '⁉️'

StratsOther = ['default']
StratsEpub = ['fname', 'title', 'html_items']
StratsPdf = ['pdf_page', 'cpn_pdf_toc']
TOCStrat = click.Choice(StratsOther+StratsEpub+StratsPdf)

PrintOnly = click.Choice(['off', 'name', 'full'])


register_type_inference(TOCStrat, TOCStrat)
register_type_inference(PrintOnly, PrintOnly)


@dataclass
class SplitArgs:
    infile: Annotated[Path, option()]
    outpath: Annotated[Path, option(default=None)]
    toc_strat: Annotated[TOCStrat, option(default='default')]
    unspace: Annotated[bool, option(default=False)]
    rm_linearization: Annotated[bool, option(default=False)]


@dataclass
class RunArgs:
    infile: Annotated[Path, option()]
    outpath: Annotated[Path, option()]
    piper_exe_path: Annotated[Path, option()]
    model_file_path: Annotated[Path, option()]
    toc_strat: Annotated[TOCStrat, option(default='default')]
    unspace: Annotated[bool, option(default=False)]
    bilingual: Annotated[bool, option(default=False)]
    print_only: Annotated[PrintOnly, option(default='off')]
    rm_linearization: Annotated[bool, option(default=False)]

def elide_text(string, max):
    trunc = re.sub('\n+\\W*', '⮐', string)
    trunc = trunc.replace('\x0c', '')
    if len(trunc) <= max:
        return trunc
    trunc = trunc[0:max-1] + "…"
    return trunc


class TTSTrack(BaseModel):
    text: str = ""
    title: str | None = None


    def __str__(self):
        trunc1 = elide_text(self.title, 48)
        trunc2 = elide_text(self.text, 48)
        #s0 = f"TTSTrack({trunc1}, {trunc2})"
        text0 = re.sub('\n', f'{sentence_char}\n', self.text)
        s0 = ""
        s0 += '#####################################################\n'
        s0 += f"TITLE: '{trunc1}'\n"
        s0 += '#####################################################\n'
        s0 += text0 + ('\n' if text0 and text0[-1] != '\n' else '')
        s0 += '####################################################'
        return s0

    def __repr__(self):
        return self.__str__()
