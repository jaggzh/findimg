# findimg.py
# jaggz.h is at gmail.com

# Starting:
# 1. Create global set of images:
#    You need to create dat/globfiles.txt
#    It's a list of all images in your global dataset
#    Ex: find /path/to/images -iname '*.jpg' > dat/globfiles.txt
# 2. Make folders of special labeled categories:
#    Ex's: imgdat/labeled/dogs
#          imgdat/labeled/frontdoors
#    Labeled folders are walked through, of any depth.
#    * DOES NOT TEST EXTENSIONS OR FILETYPES, it assumes they're
#      all images. However, it doesn't choke on them -- just
#      displays an error, removes them from the internal list,
#      and continues.

from os.path import join

# dir_labeled_base:
#  Contains folders of images, each main folder is a label
dir_labeled_base="imgdat/labeled"

# dir_cache:
#  Cached small versions of all images are generated
#   and placed here as {ha}/{hash}.jpg
#  Can get messy in here if you're testing many sets of images.
#  Current cached size is 64x64 (see indim in findimg.py)
dir_cache="cache"

# dir_dat:
#  Weight file is stored here, and you need to
#  create data/globfiles.txt (see setting file_pics_glob_files)
dir_dat="dat"

# file_pics_glob_files:
#  List of all (non-categorized) images.
#  One image (full or relative path) per line.
#  This is the 'global' label
file_pics_glob_files=dir_dat + "/globfiles.txt"

# Unused:
file_pics_glob_dirs=dir_dat + "/globdirs.txt"
dir_pics_everything="/path/to/photos"

# vim:ts=4 ai
