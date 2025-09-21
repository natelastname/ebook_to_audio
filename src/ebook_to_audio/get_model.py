#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-06-25T18:37:14-04:00

@author: nate
"""
import os
from pathlib import Path

import requests
from loguru import logger
from platformdirs import user_cache_dir
from tqdm import tqdm


def get_model(outfile_name, download_url: str) -> Path:
    outpath = user_cache_dir("ebook_to_audio")
    outpath = os.path.join(str(outpath), 'models')
    os.makedirs(outpath, exist_ok=True)
    outfile = os.path.join(outpath, outfile_name)
    model_path = Path(user_cache_dir("ebook_to_audio"))
    model_path = model_path / "models" / outfile_name

    if model_path.exists():
        logger.info(f"Using cached model at {model_path}")
        return

    logger.info(f"Downloading '{outfile_name}'...")

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(outfile, 'wb') as f, tqdm(
            desc=outfile_name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    return outfile


def auto_download():
    weights_url = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx?download=true'
    config_url = 'https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json?download=true.json'
    path1 = get_model('en_GB-alan-medium.onnx', weights_url)
    path2 = get_model('en_GB-alan-medium.onnx.json', config_url)
    return path1
