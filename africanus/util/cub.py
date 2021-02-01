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

_cub_dir = pjoin(include_dir, "cub")
_cub_url = 'https://github.com/NVlabs/cub/archive/1.8.0.zip'
_cub_sha_hash = '836f523a34c32a7e99fba36b30abfe7a68d41d4b'
_cub_version_str = 'Current release: v1.8.0 (02/16/2018)'
_cub_version = "1.8.0"
_cub_zip_dir = 'cub-' + _cub_version
_cub_download_filename = "cub-" + _cub_version + ".zip"
_cub_header = pjoin(_cub_dir, 'cub', 'cub.cuh')
_cub_readme = pjoin(_cub_dir, 'README.md')
_cub_new_unzipped_path = _cub_dir


log = logging.getLogger(__name__)


class InstallCubException(Exception):
    pass


def download_cub(archive_file):
    response = urlopen(_cub_url)

    if not os.path.exists(downloads_dir):
        os.makedirs(downloads_dir)

    with open(archive_file, "wb") as f:
        shutil.copyfileobj(response, f)

    return sha_hash_file(archive_file)


def is_cub_installed(readme_filename, header_filename, cub_version_str):
    # Check if the cub.h exists
    if (not os.path.exists(header_filename) or
            not os.path.isfile(header_filename)):

        reason = "CUB header '{}' does not exist".format(header_filename)
        return (False, reason)

    # Check if the README.md exists
    if (not os.path.exists(readme_filename) or
            not os.path.isfile(readme_filename)):

        reason = "CUB readme '{}' does not exist".format(readme_filename)
        return (False, reason)

    # Search for the version string, returning True if found
    with open(readme_filename, 'r') as f:
        for line in f:
            if line.find(cub_version_str) != -1:
                return (True, "")

    # Nothing found!
    reason = "CUB version string '{}' not found in '{}'".format(
        cub_version_str, readme_filename)

    return (False, reason)


def _install_cub():
    archive = pjoin(downloads_dir, _cub_download_filename)
    should_download = True

    # Check existing archive in the download directory
    # If the hash matches we can avoid downloading
    if os.path.exists(archive) and os.path.isfile(archive):
        if _cub_sha_hash == sha_hash_file(archive):
            should_download = False

    # Download and check the hash
    if should_download:
        sha_hash = download_cub(archive)
        # Compare against our supplied hash
        if _cub_sha_hash != sha_hash:
            msg = ('Hash of file %s downloaded from %s '
                   'is %s and does not match the expected '
                   'hash of %s.') % (
                        _cub_download_filename, _cub_url,
                        _cub_sha_hash, sha_hash)

            raise InstallCubException(msg)

    # Unzip into include/cub
    with ZipFile(archive, 'r') as zip_file:
        # Remove any existing install
        try:
            shutil.rmtree(_cub_dir, ignore_errors=True)
        except Exception as e:
            raise InstallCubException("Removing %s failed\n%s" % (
                                      _cub_dir, str(e)))

        try:
            # Unzip into temporary directory
            tmpdir = tempfile.mkdtemp()
            zip_file.extractall(tmpdir)

            unzip_path = pjoin(tmpdir, _cub_zip_dir)

            # Move
            shutil.move(unzip_path, _cub_dir)
        except Exception as e:
            raise InstallCubException("Extracting %s failed\n%s" % (
                                      archive, str(e)))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        log.info("NVIDIA cub archive unzipped into '%s'" % _cub_dir)

    # Final check on installation
    there, reason = is_cub_installed(_cub_readme, _cub_header,
                                     _cub_version_str)

    if not there:
        raise InstallCubException(reason)


_cub_install_lock = Lock()

with _cub_install_lock:
    _cub_installed, _ = is_cub_installed(_cub_readme, _cub_header,
                                         _cub_version_str)


def cub_dir():
    global _cub_installed

    if _cub_installed is False:
        with _cub_install_lock:
            # Double-locking pattern
            if _cub_installed is False:
                _install_cub()
                _cub_installed = True

    return _cub_dir
