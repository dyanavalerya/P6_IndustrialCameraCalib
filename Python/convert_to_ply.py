# This requires CloudCompare software installed with flatpak
# I used Version 2.12.0 (Kyiv), compiled with Qt 5.15.3

import os
import subprocess 

# get the current working directory
dir_path = os.getcwd()

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if path != 'convert_to_ply.py':
        if os.path.isfile(os.path.join(dir_path, path)):
            os.system('flatpak run org.cloudcompare.CloudCompare -SILENT '
                         '-O ' + str(path) + ' -OCTREE_NORMALS 10 -C_EXPORT_FMT PLY -SAVE_CLOUDS')
