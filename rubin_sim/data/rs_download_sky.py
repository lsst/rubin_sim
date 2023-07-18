__all__ = ("MyHTMLParser", "rs_download_sky")

import argparse
import os
from html.parser import HTMLParser

import requests

from . import get_data_dir


# Hack it up to find the filenames ending with .h5
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        try:
            self.filenames
        except AttributeError:
            setattr(self, "filenames", [])
        if tag == "a":
            if attrs[0][0] == "href":
                if attrs[0][1].endswith(".h5"):
                    self.filenames.append(attrs[0][1])


def rs_download_sky():
    """Download sky files."""

    parser = argparse.ArgumentParser(
        description="Download precomputed skybrightness files for rubin_sim package"
    )
    parser.add_argument(
        "-f",
        "--force",
        dest="force",
        default=False,
        action="store_true",
        help="Force re-download of sky brightness data.",
    )
    parser.add_argument(
        "--url_base",
        type=str,
        default="https://s3df.slac.stanford.edu/groups/rubin/static/sim-data/sims_skybrightness_pre/h5/",
        help="Root URL of download location",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    destination = os.path.join(data_dir, "skybrightness_pre")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    if not os.path.isdir(destination):
        os.mkdir(destination)

    # Get the index file
    r = requests.get(args.url_base)
    # Find the filenames
    parser = MyHTMLParser()
    parser.feed(r.text)
    parser.close()
    # Copy the sky data files, if they're not already present
    for file in parser.filenames:
        if not os.path.isfile(os.path.join(destination, file)) or args.force:
            url = args.url_base + file
            print(f"Downloading file {file} from {url}")
            r = requests.get(url)
            with open(os.path.join(destination, file), "wb") as f:
                f.write(r.content)
