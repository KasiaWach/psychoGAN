import sys
import os
import numpy as np
import pandas as pd
import PIL.Image
from PIL import Image, ImageDraw
import imageio
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd()+"/stylegan2")
from generator import generator
def main():
    main_generator = generator(network_pkl_path="gdrive:networks/stylegan2-ffhq-config-f.pkl",
                               direction_path="stylegan2/stylegan2directions/dominance.npy", coefficient=1.0,
                               truncation=0.7, n_levels=3, n_photos=10, type_of_preview="manipulation",
                               result_dir="/results")
    plt.imshow(generator._generator__generate_preview_face_manip())


main()