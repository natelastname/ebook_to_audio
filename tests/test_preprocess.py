# -*- coding: utf-8 -*-
"""
Created on 2025-04-13T20:03:31-04:00

@author: nate
"""

import ebook_to_audio as e2a


def test_yank_re():
    yanked_from, yanked = e2a.preprocess.yank_re('[0-9]+', 'asda 1231 daasad 1231 j1j1j1')
    assert yanked_from == 'asda  daasad  jjj'
    assert yanked == '12311231111'
