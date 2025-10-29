#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-01-12T08:30:13-05:00

@author: nate
"""
import atexit
import datetime
import functools
import gzip
import os
import re
import string
import subprocess as sp
import sys
import tempfile
import unicodedata
from urllib.parse import urlparse

import ebook_to_audio as e2a
import enchant
import hakilo
import omterms
import popplerqt5
import readtime
import wordninja
from ebook_to_audio import types
from transliterate import translit
from unidecode import unidecode


def fix_spacing(text):
    # Match Unicode and non-Unicode quotes
    quote_pattern = r'[\'"\u2018\u2019\u201C\u201D]'
    # Fix spacing: remove spaces before quotes, ensure one space after
    # Step 1: Remove spaces before quotes (e.g., "dog ," → "dog,")
    text = re.sub(r'\s+(' + quote_pattern + r')', r'\1', text)
    # Step 2: Ensure one space after quotes when followed by a non-space (e.g., ","said" → "," said")
    text = re.sub(r'(' + quote_pattern + r')(\S)', r'\1 \2', text)
    # Step 3: Normalize multiple spaces after quotes to one (e.g., ","  said" → "," said")
    text = re.sub(r'(' + quote_pattern + r')\s+', r'\1 ', text)
    text = re.sub(r',(\S)', r', \1', text)
    return text


class Unspacer():
    def __init__(self):
        self.word_file = os.path.dirname(os.path.abspath(wordninja.__file__))
        self.word_file = os.path.join(self.word_file, 'wordninja','wordninja_words.txt.gz')
        with gzip.open(self.word_file) as f:
            self.words = f.read().decode().split()

        self.letters = set()
        for word in self.words:
            word_letters = set([l0 for l0 in word])
            self.letters = self.letters.union(word_letters)

        self.letters = list(self.letters)

    def _toke(self, text0: str):
        if not text0:
            return text0
        flag = False
        if text0[0] in self.letters:
            flag = True
        out_text = ""
        for letter in text0:

            # If the letter is in the corpus or it is whitespace,
            cond1 = letter.lower() in self.letters
            cond1 = cond1 or re.match('\\s', letter)

            if not cond1 and flag == False:
                # Continue non-char block
                out_text += letter
                continue
            elif not cond1 and flag == True:
                # char block -> non-char block
                yield flag, out_text
                out_text = letter
                flag = False
                continue
            elif cond1 and flag == False:
                # non-char block -> char block
                yield flag, out_text
                out_text = letter
                flag = True
                continue
            elif cond1 and flag == True:
                # non-char block -> char block
                out_text += letter
                continue
            raise Exception(f'unhandled case: "{letter}", {flag} ')
        yield flag, out_text

    def unspace(self, text0: str):
        terms = omterms.interface.extract_terms(text0)
        terms = [t0 for t0 in terms['Term'].unique()]

        self.words = list(set(self.words).union(set(terms)))

        out_text = ""
        for i, (flag, block) in enumerate(self._toke(text0)):
            if flag:
                block = wordninja.split(''.join(block.split()))
                block = ' '.join(block)
            else:
                block += ''
            #print(f'{i:5}: "{block}", {flag}')
            out_text = out_text + block
        out_text = re.sub("\\s*'\\s*", "'", out_text)
        #re.search(pattern, text, re.IGNORECASE)
        #out_text = fix_spacing(out_text.strip())
        return out_text.strip()



######################################################################
# Text conversion / processing
######################################################################

def yank_re(re0, str0):
    out0 = ""
    for m0 in re.finditer(re0, str0):
        out0 += m0.group()
    out1 = re.sub(re0, '', str0)
    return out1, out0


def gen_basename(infile):
    #basename_orig = os.path.basename(infile)
    #basename_orig = re.sub('[^A-Za-z0-9]+', '', basename_orig)
    basename = re.sub("\\W", '', infile)
    basename = re.sub('\\_', '', basename)
    m0 = re.match('(\\w+)', basename)
    basename = m0.groups()[0]
    return basename

def _unidecode(txt):
    txt, n = re.subn(types.sentence_char, '', txt)
    txt = unidecode(txt)
    return txt + (types.sentence_char*n)

def split_hakilo(txt):
    out_lines = []
    found = False
    for line in hakilo.split_text(txt):
        line = re.sub('[\r\n\f]+', ' ', line)
        yield line
    return
    #return out_lines



def hakilo_sentences(text):
    page_num = 0
    page = ""
    gen0 = e2a.preprocess.split_hakilo(text)
    for i, sentence in enumerate(gen0):
        yanked_from, yanked = yank_re(f'[{e2a.types.page_char}]', sentence)
        line = yanked_from+yanked
        yield line

def pp_remove_newlines(text):
    filtered = re.sub('(\\s*\n)+', '\n', text)
    return filtered

def pp_remove_watermarks(text):
    filtered = re.sub('\\*+ebook converter DEMO Watermarks\\*+', '', text)
    return filtered

def pp_truncate_text(text):
    res = text.split('\n')[0:300]
    return "\n".join(res)

def pp_break_paragraphs(text):
    res = text.split('\n')
    newlines = []
    for line in res:
        temp = re.sub('[1-9]', '', line)
        if temp.strip().endswith(('.', ':')):
            line = line + "\n"
        newlines.append(line)
    text = "\n".join(newlines)
    text = re.split('\\s*\n\\s*\n', text)
    newlines = []
    for line in text:
        newline = re.sub('\n', ' ', line)
        newlines.append(newline)
    text = '\n\n'.join(newlines)
    return text

def pp_fix_formfeeds(text):

    # Pre-compile for speed if you call this a lot
    _SOFT_OR_ZERO_WIDTH = re.compile(r'[\u00AD\u200B\u200C\u200D]')  # soft hyphen & zero-widths
    # Hyphen-like chars that are used for word breaks (not em/en dashes)
    _WORD_HYPHEN = r'[-\u2010\u2011\u00AD]'  # -, hyphen, non-breaking hyphen, soft hyphen
    # 1) Join hyphenated words that break across newline or form-feed
    _JOIN_HYPHEN_BREAK = re.compile(
        rf'(\w){_WORD_HYPHEN}\s*(?:\r?\n|\r|\x0c)\s*(?=[A-Za-z])'
    )
    # 2) Replace any whitespace + form-feed + whitespace with a paragraph break
    _FORMFEED_TO_PARA = re.compile(r'\s*\x0c+\s*')
    # 3) Normalize multiple blank lines and excess spaces
    _MULTIBLANKS = re.compile(r'\n{3,}')
    _MULTISPACES = re.compile(r'[ \t]{2,}')
    # Normalize line endings early
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove soft/zero-width characters that confuse tokenization/tts
    text = _SOFT_OR_ZERO_WIDTH.sub('', text)
    # Join hyphenated words that were split by newline or form-feed:
    # "inter-\n national" -> "international", "co-\x0c operate" -> "cooperate"
    text = _JOIN_HYPHEN_BREAK.sub(r'\1', text)
    # Convert form-feeds to paragraph breaks; this keeps a pause for TTS
    text = _FORMFEED_TO_PARA.sub('\n\n', text)
    # Collapse silly spacing
    text = _MULTIBLANKS.sub('\n\n', text)
    text = _MULTISPACES.sub(' ', text)

    return text

    # Optional: ensure space around em/en dashes so TTS pauses naturally
    # text = re.sub(r'\s*([\u2012\u2013\u2014])\s*', r' \1 ', text)



    # Remove line-end hyphens
    text = re.sub('\\S[—-]\\s+\\S', '', text)
    # if the page break breaks a line, delete all whitespace
    text = re.sub('-\\s*\x0c\\s*', '\x0c', text)
    # If the page break is at a word boundary, leave a space
    text = re.sub('-?\\s*\x0c\\s*', ' \x0c', text)
    breakpoint()
    return text


def pp_expand_honorifics(text):
    # OPening parens, quotes..
    text = re.sub('\\s+Mr\\.', ' Mister', text)
    text = re.sub('\\s+Mrs\\.', ' Misses', text)
    text = re.sub('\\s+Ms\\.', ' Miss', text)
    text = re.sub('\\s+Fr\\.', ' Father', text)
    text = re.sub('\\s+St\\.', ' Saint', text)
    text = re.sub('\\s+Pvt\\.', ' Private', text)
    text = re.sub('\\s+Sgt\\.', ' Sargeant', text)
    text = re.sub('\\s+Lt\\.', ' Lieutenant', text)
    text = re.sub('\\s+Capt\\.', ' Captain', text)
    text = re.sub('\\s+Col\\.', ' Colonel', text)
    text = re.sub('\\s+Adm\\.', ' Admiral', text)
    text = re.sub('\\s+Rev\\.', ' Reverend', text)
    return text


def enchant_sentence(sentence: str) -> str:
    # Currently unused
    d = enchant.Dict("en_US")
    words = sentence.split()
    fixed_text = []
    for i in range(len(words)):
        if fixed_text and not d.check(words[i]):
            compound = fixed_text[-1] + words[i]
            if d.check(compound):
                fixed_text[-1] = compound
            else:
                fixed_text.append(words[i])
        else:
            fixed_text.append(words[i])
    return " ".join(fixed_text)


def filter_non_printable(s0: str):
    chars = []
    filtered = ['Cc']
    for char0 in s0:
        cat = unicodedata.category(char0)
        if cat in filtered:
            breakpoint()
            continue
        chars.append(char0)
    out_str = ''.join(chars)
    return out_str

def do_translit(text):
    found = {}
    for char in text:
        if unicodedata.name(char, "").startswith("GREEK"):
            found['el'] = True
        if unicodedata.name(char, "").startswith("CYRILLIC"):
            found['ru'] = True
    for key, val in found.items():
        text = translit(text, key, reversed=True)
    return text

def pp_normalize_whitespace(text):
    chars0 = string.whitespace.replace('\n', '')
    #chars0 += '\xad'
    text = re.sub(f"[{chars0}]+", ' ', text)
    return text

def preprocess_text(text, unspace: bool):

    str0 = '\xc2\xad'
    text = re.sub(f"[{str0}]", '', text)
    '''
    for char in str0:
        if text.find(char) >= 0:
            breakpoint()
    '''
    text = re.sub('[\x0c]', e2a.types.page_char, text)

    out_pages = []
    # Split by page because the page boundary is meaningful
    text = pp_remove_newlines(text)
    text = pp_break_paragraphs(text)
    text = pp_fix_formfeeds(text)
    text = pp_expand_honorifics(text)
    #text = pp_normalize_whitespace(text)

    gen0 = hakilo_sentences(text)
    result = ""
    unspacer = Unspacer()
    for sentence in gen0:
        sent = filter_non_printable(sentence)
        if sent.strip() == "":
            continue
        s2 = do_translit(sent)
        if unspace:
            s2 = unspacer.unspace(s2)
        result += s2 + "\n"
    return result


def preprocess_text_lite(text, unspace: bool):
    str0 = '\xc2\xad'
    text = re.sub(f"[{str0}]", '', text)
    text = re.sub('[\x0c]', e2a.types.page_char, text)


    '''
    word_file = os.path.dirname(os.path.abspath(wordninja.__file__))
    word_file = os.path.join(word_file, 'wordninja','wordninja_words.txt.gz')
    with gzip.open(word_file) as f:
        words = f.read().decode().split()
    terms = omterms.interface.extract_terms(text)
    terms = [t0 for t0 in terms['Term'].unique()]
    words = list(set(words).union(set(terms)))
    '''
    #breakpoint()
    '''
    text = re.sub('[^\S\n]+', ' ', text)
    text = re.sub('\\s+\n', '\n', text)
    text = re.sub('\n\\s+', '\n', text)
    '''
    #text = re.sub('\n+', '\n', text)

    pages = text.split(e2a.types.page_char)
    out_pages = []
    for page in pages:
        #result = e2a.preprocess.preprocess_text(page, unspace)
        result = pp_fix_formfeeds(page)
        out_pages.append(result)

    text = e2a.types.page_char.join(out_pages)

    return text
