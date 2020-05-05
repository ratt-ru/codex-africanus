# -*- coding: utf-8 -*-


import logging
import os
from os.path import join as pjoin
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock

from africanus.util.appdirs import downloads_dir, include_dir
from africanus.util.files import sha_hash_file

_trove_dir = pjoin(include_dir, "trove")
_trove_url = 'https://github.com/bryancatanzaro/trove/archive/master.zip'
_trove_sha_hash = '183c9ce229b3c0b2afeb808a9f4b07c9d9b9035d'
_trove_version_str = 'Current release: v1.8.0 (02/16/2018)'
_trove_version = "master"
_trove_zip_dir = 'trove-' + _trove_version
_trove_download_filename = "trove-" + _trove_version + ".zip"
_trove_header = pjoin(_trove_dir, 'trove', 'trove.cuh')
_trove_readme = pjoin(_trove_dir, 'README.md')
_trove_new_unzipped_path = _trove_dir


log = logging.getLogger(__name__)


class InstallTroveException(Exception):
    pass


def download_trove(archive_file):
    response = urlopen(_trove_url)

    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)

    with open(archive_file, "wb") as f:
        shutil.copyfileobj(response, f)

    return sha_hash_file(archive_file)


def is_trove_installed(readme_filename):
    # Check if the README.md exists
    if (not os.path.exists(readme_filename) or
            not os.path.isfile(readme_filename)):

        reason = "trove readme '{}' does not exist".format(readme_filename)
        return (False, reason)

    return (True, "")


def _install_trove():
    archive = pjoin(downloads_dir, _trove_download_filename)
    should_download = True

    # Check existing archive in the download directory
    # If the hash matches we can avoid downloading
    if os.path.exists(archive) and os.path.isfile(archive):
        if _trove_sha_hash == sha_hash_file(archive):
            should_download = False

    # Download and check the hash
    if should_download:
        sha_hash = download_trove(archive)
        # Compare against our supplied hash
        if _trove_sha_hash != sha_hash:
            msg = ('Hash of file %s downloaded from %s '
                   'is %s and does not match the expected '
                   'hash of %s.') % (
                        _trove_download_filename, _trove_url,
                        sha_hash, _trove_sha_hash)

            raise InstallTroveException(msg)

    # Unzip into include/trove
    with ZipFile(archive, 'r') as zip_file:
        # Remove any existing install
        try:
            shutil.rmtree(_trove_dir, ignore_errors=True)
        except Exception as e:
            raise InstallTroveException("Removing %s failed\n%s" % (
                                      _trove_dir, str(e)))

        try:
            # Unzip into temporary directory
            tmpdir = tempfile.mkdtemp()
            zip_file.extractall(tmpdir)

            unzip_path = pjoin(tmpdir, _trove_zip_dir)

            # Move
            shutil.move(unzip_path, _trove_dir)
        except Exception as e:
            raise InstallTroveException("Extracting %s failed\n%s" % (
                                      archive, str(e)))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        log.info("NVIDIA trove archive unzipped into '%s'" % _trove_dir)

    # Final check on installation
    there, reason = is_trove_installed(_trove_readme)

    if not there:
        raise InstallTroveException(reason)


_trove_install_lock = Lock()

with _trove_install_lock:
    _trove_installed, _ = is_trove_installed(_trove_readme)


def trove_dir():
    global _trove_installed

    if _trove_installed is False:
        with _trove_install_lock:
            # Double-locking pattern
            if _trove_installed is False:
                _install_trove()
                _trove_installed = True

    return _trove_dir
