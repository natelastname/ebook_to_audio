#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-03-16T19:53:19-04:00

@author: nate
"""

import re
from copy import copy
from typing import Callable

import bs4
import ebooklib
import pandas as pd

from ebook_to_audio.types import RunArgs


def html_split(html_str, group_name, ids):
    if not isinstance(html_str, bs4.BeautifulSoup):
        soup = bs4.BeautifulSoup(html_str, 'html.parser')
    else:
        soup = html_str

    skele = copy(soup)
    skele.find('body').clear()

    body = soup.body
    item = body
    top_level = "init"
    text = ""

    res = []
    while True:
        item = item.next_element
        if not item:
            yield top_level, top_level, text
            break

        if isinstance(item, bs4.NavigableString):
            text += item + "\n"
            continue
        id0 = item.attrs.get('id')
        if not id0 in ids:
            continue

        with_frag = f"{group_name}"
        if top_level:
            with_frag += f"#{top_level}"


        out_name = with_frag
        if re.match('h[0-9]+', item.name):
            out_name = item.text

        yield out_name, with_frag, text

        top_level = id0
        text = ""

    return res
