# -*- coding: utf-8 -*-


from hashlib import sha1


def sha_hash_file(filename):
    """ Compute the SHA1 hash of filename """
    hash_sha = sha1()

    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            hash_sha.update(chunk)

    return hash_sha.hexdigest()
