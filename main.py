import csv
import numpy as np
import pandas as pd
import PIL.Image
from PIL import Image, ImageDraw
# import dnnlib
# import dnnlib.tflib as tflib
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
import pretrained_networks

class generator():
    def __init__(self, network_pkl_path,direction_path,coefficient,truncation,n_levels,n_photos,type_of_preview,result_dir):
        self.direction_path = direction_path    # Ścieżka do wektora cechy
        self.direction = np.load(direction_path)# Wgrany wektor cechy
        self.coefficient = coefficient          # Siła manipluacji / przemnożenie wektora
        self.truncation = truncation            # Parametr stylegan "jak różnorodne twarze"
        self.n_levels = n_levels                # liczba poziomów manipulacji 1-3
        self.n_photos = n_photos                # Ile zdjęć wygenerować
        self.preview_face = np.array([])        # Array z koordynatami twarzy na podglądzie 1
        self.preview_3faces = np.array([])      # Array z koordynatami twarzy na podglądzie 3
        self.synthesis_kwargs = {}              # Keyword arguments które przyjmuje stylegan
        self.type_of_preview = type_of_preview  # Typ podglądu, wartości: "3_faces", "manipulation" w zależności od tego które ustawienia są zmieniane
        self.result_dir = result_dir
        self._G, self._D, self.Gs = pretrained_networks.load_networks(network_pkl_path)

    def refresh_preview(self):
        pass

    def change_face(self):
        all_z = np.random.randn(1, *Gs.input_shape[1:])
        all_w = self.__map_vectors(all_z)
        all_w = self.__truncate_vectors(all_w)

    def generate(self):
        pass

    def __generate_preview_face_manip(self):
        """Zwraca PIL Image ze sklejonymi 3 twarzami w środku neutralna, po bokach zmanipulowana"""
        all_w = self.__tile_vector(self.preview_face)  #Rozwinęcie wektor w 18 razy i wrzucenie na liste
        # generewanie losowej/losowych twarzy

        all_w = np.array([all_w[0],all_w[0],all_w[0]])  # Przygotowujemy miejsca na twarze zmanipulowane

         # przesunięcie twarzy o wektor (już rozwinięty w 18)
        all_w[0][0:8] = (all_w[0] + self.coefficient * self.direction)[0:8]
        all_w[2][0:8] = (all_w[2] - self.coefficient * self.direction)[0:8]

        all_images = Gs.components.synthesis.run(all_w, **self.synthesis_kwargs)
        images = [PIL.Image.fromarray(all_images[i], 'RGB') for i in range(3)]

        return PIL.Image.hstack(images)

    def __tile_vector(self, faces_w):
        """Przyjmuje listę 512-wymierowych wektorów twarzy i rozwija je w taki które przyjmuje generator"""
        return np.array([np.tile(face, (18, 1)) for face in faces_w])

    def __generate_preview_face_face_3(self):
        """__generate_preview_face_manip tylko że używa zmiennej preview_3faces zamist preview_face"""
         pass

    def __map_vectors(self, faces_z):
         """Przyjmuje array wektorów z koordynatami twarzy w Z-space, gdzie losowane są wektory,
         zwraca array przerzucony do w-space, gdzie dzieje się manipulacja"""
         return self.Gs.components.mapping.run(faces_z, None)

    def __truncate_vectors(self, faces_w):
        """Zwraca wektory z faces_w przesunięte w kierunku uśrednionej twarzy"""
        w_avg = self.Gs.get_var('dlatent_avg')
        return w_avg + (faces_w - w_avg) * self.truncation

    def __kwargs(self):
        Gs_syn_kwargs = dnnlib.EasyDict()
        Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                              nchw_to_nhwc=True)
        Gs_syn_kwargs.randomize_noise = False
        Gs_syn_kwargs.minibatch_size = 1


def main():
    generator = generator(network_pkl_path="",direction_path="stylegan2/stylegan2directions/dominance.npy",coefficient=1.0,truncation=0.7,n_levels=3,n_photos=10,type_of_preview="manipulation",result_dir="/results")
    plt.imshow(generator.__generate_preview_face_manip())


main()