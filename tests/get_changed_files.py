"""Hashes midi files to see if they are new or have changed. If so, we 
re-generate the pngs therefrom.
"""

# after https://stackoverflow.com/a/36113168/10155119
from collections import defaultdict
import hashlib
import os
import sys
import json
from typing import Sequence


def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def get_hash(filename, first_chunk_only=False, hash=hashlib.sha1):
    hashobj = hash()
    file_object = open(filename, "rb")

    if first_chunk_only:
        hashobj.update(file_object.read(1024))
    else:
        for chunk in chunk_reader(file_object):
            hashobj.update(chunk)
    hashed = hashobj.hexdigest()

    file_object.close()
    return hashed


def get_changed_files(
    mem_id: str,
    basenames: Sequence[str],
    dirname: str,
    return_memory_updater=True,
):
    """
    mem_id: str. Arbitrary string to identify the memory file that will be
        saved.
    """

    file_mem_path = os.path.join(
        os.path.dirname((os.path.realpath(__file__))),
        f".{mem_id}_file_mem.json",
    )

    TEST_OUT_DIR = os.path.join(
        os.path.dirname((os.path.realpath(__file__))), "test_out"
    )
    if not os.path.exists(TEST_OUT_DIR):
        os.makedirs(TEST_OUT_DIR)

    if not os.path.exists(file_mem_path):
        file_mem = {}
    else:
        with open(file_mem_path, "r") as inf:
            file_mem = json.load(inf)

    file_mem = {f: v for f, v in file_mem.items() if f in basenames}
    out = []
    for basename in basenames:
        full_path = os.path.join(dirname, basename)
        if basename in file_mem:
            attrs = file_mem[basename]
            size = os.path.getsize(full_path)
            if size == attrs["size"]:
                small_hash = get_hash(full_path, first_chunk_only=True)
                if small_hash == attrs["small_hash"]:
                    full_hash = get_hash(full_path, first_chunk_only=False)
                    if full_hash == attrs["full_hash"]:
                        continue
        else:
            size = os.path.getsize(full_path)
            small_hash = get_hash(full_path, first_chunk_only=True)
            full_hash = get_hash(full_path, first_chunk_only=False)
            file_mem[basename] = {
                "size": size,
                "small_hash": small_hash,
                "full_hash": full_hash,
            }
        out.append(basename)

    out = [os.path.join(dirname, basename) for basename in out]
    if return_memory_updater:

        def _memory_updater():
            with open(file_mem_path, "w") as outf:
                json.dump(file_mem, outf)

        return out, _memory_updater
    return out


# def check_for_duplicates(paths, hash=hashlib.sha1):
#     hashes_by_size = defaultdict(
#         list
#     )  # dict of size_in_bytes: [full_path_to_file1, full_path_to_file2, ]
#     hashes_on_1k = defaultdict(
#         list
#     )  # dict of (hash1k, size_in_bytes): [full_path_to_file1, full_path_to_file2, ]
#     hashes_full = {}  # dict of full_file_hash: full_path_to_file_string

#     for path in paths:
#         for dirpath, dirnames, filenames in os.walk(path):
#             # get all files that have the same size - they are the collision candidates
#             for filename in filenames:
#                 full_path = os.path.join(dirpath, filename)
#                 try:
#                     # if the target is a symlink (soft one), this will
#                     # dereference it - change the value to the actual target file
#                     full_path = os.path.realpath(full_path)
#                     file_size = os.path.getsize(full_path)
#                     hashes_by_size[file_size].append(full_path)
#                 except (OSError,):
#                     # not accessible (permissions, etc) - pass on
#                     continue

#     # For all files with the same file size, get their hash on the 1st 1024 bytes only
#     for size_in_bytes, files in hashes_by_size.items():
#         if len(files) < 2:
#             continue  # this file size is unique, no need to spend CPU cycles on it

#         for filename in files:
#             try:
#                 small_hash = get_hash(filename, first_chunk_only=True)
#                 # the key is the hash on the first 1024 bytes plus the size - to
#                 # avoid collisions on equal hashes in the first part of the file
#                 # credits to @Futal for the optimization
#                 hashes_on_1k[(small_hash, size_in_bytes)].append(filename)
#             except (OSError,):
#                 # the file access might've changed till the exec point got here
#                 continue

#     # For all files with the hash on the 1st 1024 bytes, get their hash on the full file - collisions will be duplicates
#     for __, files_list in hashes_on_1k.items():
#         if len(files_list) < 2:
#             continue  # this hash of fist 1k file bytes is unique, no need to spend cpy cycles on it

#         for filename in files_list:
#             try:
#                 full_hash = get_hash(filename, first_chunk_only=False)
#                 duplicate = hashes_full.get(full_hash)
#                 if duplicate:
#                     print(
#                         "Duplicate found: {} and {}".format(filename, duplicate)
#                     )
#                 else:
#                     hashes_full[full_hash] = filename
#             except (OSError,):
#                 # the file access might've changed till the exec point got here
#                 continue


# if __name__ == "__main__":
#     if sys.argv[1:]:
#         check_for_duplicates(sys.argv[1:])
#     else:
#         print("Please pass the paths to check as parameters to the script")
