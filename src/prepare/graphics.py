# -*-coding:utf-8 -*-
"""
@File      :   graphics.py
@Time      :   2025/04/23 16:53:50
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   This module extracts the graphics from the CSS file of OGame.
              It downloads the images and saves them in a specified directory.
"""

import re
import os
import shutil
import urllib
from tqdm import tqdm

from utils.paths import output, data, path_to_file


URL_PATTERN = re.compile(r"url\((.*?)\)")

BACKGROUND_PATTERN = re.compile(
    r"#technologydetails\.(?P<name>\w+)[: ]"
    r"[^\(]*background-image: url\("
    r"\"[/\w\.]+/(?P<url>\w+\.(?:png|jpg))\""
)


def extract_urls_from_css(css: str) -> list[str]:
    """
    Extracts URLs from the given CSS string.

    Parameters
    ----------
    css : str
        The CSS string to extract URLs from.

    Returns
    -------
    list[str]
        A list of extracted URLs.
    """
    urls = URL_PATTERN.findall(css)
    urls = [url.strip('"').strip("'") for url in urls]
    urls = [url if url.startswith("http") else "https:" + url for url in urls]

    return urls


def download_urls(urls: list[str], output_dir: str) -> None:
    """
    Downloads the files from the given URLs and saves them to the specified directory.

    Parameters
    ----------
    urls : list[str]
        The list of URLs to download.
    output_dir : str
        The directory to save the downloaded files.
    """
    errors = []
    for url in (pbar := tqdm(urls, total=len(urls), desc="Downloading files")):
        pbar.set_postfix_str(url)
        filename = urllib.parse.urlparse(url).path.split("/")[-1]
        output_path = output(f"{output_dir}/{filename}").absolute().as_posix()

        if os.path.exists(output_path):
            continue
        try:
            urllib.request.urlretrieve(url, output_path)
        except Exception as e:
            errors.append((url, str(e)))
            print(f"Error downloading {url}: {e}")

    return errors


def find_backgrounds_mappings(css: str) -> dict[str, str]:
    """
    Finds the mappings of technology names to their background images in the given CSS string.

    Parameters
    ----------
    css : str
        The CSS string to extract mappings from.

    Returns
    -------
    dict[str, str]
        A dictionary mapping technology names to their background image URLs.
    """
    mappings = {}
    for match in BACKGROUND_PATTERN.finditer(css):
        name = match.group("name")
        url = match.group("url")
        mappings[name] = url

    return mappings


if __name__ == "__main__":
    # Paths
    RESOURCE_FILE = data("raw/1bee2581139faeb7d6e395e6c48f91.css")
    DOWNLOAD_DIR = path_to_file("downloads/ogame/constructions")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    IMAGES_DIR = path_to_file("images/ogame")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESOURCE_FILE, "r", encoding="utf-8") as file:
        css = file.read()

    # Find URLS in the text
    urls = extract_urls_from_css(css)

    # Find the mappings of technology names to their background images
    mappings = find_backgrounds_mappings(css)

    # Download the files
    download_urls(urls, DOWNLOAD_DIR)

    # Put the wanted images in the images directory
    for name, url in mappings.items():
        filename = urllib.parse.urlparse(url).path.split("/")[-1]
        output_path = output(f"{DOWNLOAD_DIR}/{filename}").absolute().as_posix()
        new_path = IMAGES_DIR / f"{name}.png"
        if os.path.exists(output_path):
            if not os.path.exists(new_path):
                shutil.copy(output_path, new_path)
        else:
            print(f"File {output_path} does not exist.")
