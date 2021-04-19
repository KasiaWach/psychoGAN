import csv
import numpy as np
import pandas as pd
import PIL.Image
from PIL import Image, ImageDraw
import cv2
import dnnlib
import dnnlib.tflib as tflib
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
from pretrained_networks import load_networks

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
        self.result_dir = Path(result_dir)
        self._G, self._D, self.Gs = load_networks(network_pkl_path)

    def refresh_preview(self):
        """Przełączniki co wywołać w zależności od wartości type_of_preview"""
        pass

    def __create_coordinates(self, n_photos):
        all_z = np.random.randn(n_photos, *self.Gs.input_shape[1:])
        all_w = self.__map_vectors(all_z)
        return self.__truncate_vectors(all_w)


    def change_face(self):
        if self.type_of_preview == "manipulation":
            self.preview_face = self.__create_coordinates(1)
        else:
            self.preview_3faces = self.__create_coordinates(3)


    def generate(self):
        """Zapisuje wyniki, na razie n_levels=1 """
        minibatch_size = 8

        images_dir = self.result_dir / 'images'         #Można się zastanowić czy nie zrobić z tego zmiennych obiektu, bo możliwe że będziemy się do nich częściej odnosić
        dlatents_dir = self.result_dir / 'dlatents'

        images_dir.mkdir(exist_ok=True, parents=True)
        dlatents_dir.mkdir(exist_ok=True, parents=True)

        self.__set_synthesis_kwargs(minibatch_size)


        for i in range(self.n_photos // minibatch_size): # dodajmy ładowanie w interfejsie :) /tqdm był do usunięcia
            all_w = self.__create_coordinates(minibatch_size) #Tu było n_photos a powinno być minibatch_size bo pętla ma robić minibatch_size zdjęć za każdym razem

            # error handing był tu niepotrzebny, mógł wywalić program, ale jak go dobrze napiszemy nie będzie potrzeby

            pos_w = all_w.copy()        #Będzie do dodania obsługiwanie kilku poziomów
            neg_w = all_w.copy()

            for j in range(len(all_w)):
                pos_w[j][0:8] = (pos_w[j] + self.coefficient * self.direction)[0:8]
                neg_w[j][0:8] = (neg_w[j] - self.coefficient * self.direction)[0:8]

            pos_images = self.Gs.components.synthesis.run(pos_w,
                                                     **self.synthesis_kwargs)
            neg_images = self.Gs.components.synthesis.run(neg_w,
                                                     **self.synthesis_kwargs)

            for j in range(len(all_w)):
                pos_image_pil = PIL.Image.fromarray(pos_images[j], 'RGB') #Można pomyśleć nad funkcją zapisującą obraazki która będzie miała możliwość zapisywania full jakości i miniaturkowej jakości
                pos_image_pil.save(
                    images_dir / 'tr_{}_{}.png'.format(i * minibatch_size +
                                                       j, self.coefficient))

                neg_image_pil = PIL.Image.fromarray(neg_images[j], 'RGB')
                neg_image_pil.save(
                    images_dir / 'tr_{}_-{}.png'.format(i * minibatch_size +
                                                        j, self.coefficient))

            all_images = self.Gs.components.synthesis.run(all_w, **self.synthesis_kwargs)

            for j, (dlatent, image) in enumerate(zip(all_w, all_images)):
                image_pil = PIL.Image.fromarray(image, 'RGB')
                image_pil.save(images_dir / (str(i * minibatch_size + j) + '.png'))
                np.save(dlatents_dir / (str(i * minibatch_size + j) + '.npy'),
                        dlatent[0])

    def __generate_preview_face_manip(self):
        """Zwraca array ze zdjeciem, sklejonymi 3 twarzami: w środku neutralna, po bokach zmanipulowana"""
        self.__set_synthesis_kwargs(minibatch_size=3)
        all_w = self.preview_face.copy()

        all_w = np.array([all_w[0],all_w[0],all_w[0]])  # Przygotowujemy miejsca na twarze zmanipulowane

        # Przesunięcie twarzy o wektor (już rozwinięty w 18)
        all_w[0][0:8] = (all_w[0] - self.coefficient * self.direction)[0:8]
        all_w[2][0:8] = (all_w[2] + self.coefficient * self.direction)[0:8]

        all_images = self.Gs.components.synthesis.run(all_w, **self.synthesis_kwargs)

        return np.hstack(all_images)

    def __generate_preview_face_face_3(self):
        """__generate_preview_face_manip tylko że używa zmiennej preview_3faces zamiast preview_face"""
        self.__set_synthesis_kwargs(minibatch_size=3)
        all_w = self.preview_3faces.copy()

        all_w = np.array([all_w[0],all_w[0],all_w[0]])  # Przygotowujemy miejsca na twarze zmanipulowane

        # Przesunięcie twarzy o wektor (już rozwinięty w 18)
        all_w[0][0:8] = (all_w[0] - self.coefficient * self.direction)[0:8]
        all_w[2][0:8] = (all_w[2] + self.coefficient * self.direction)[0:8]

        all_images = self.Gs.components.synthesis.run(all_w, **self.synthesis_kwargs)

        return np.hstack(all_images)

    def __tile_vector(self, faces_w):
        """Przyjmuje listę 512-wymierowych wektorów twarzy i rozwija je w taki które przyjmuje generator"""
        return np.array([np.tile(face, (18, 1)) for face in faces_w])

    def __map_vectors(self, faces_z):
         """Przyjmuje array wektorów z koordynatami twarzy w Z-space, gdzie losowane są wektory,
         zwraca array przerzucony do w-space, gdzie dzieje się manipulacja"""
         return self.Gs.components.mapping.run(faces_z, None)

    def __truncate_vectors(self, faces_w):
        """Zwraca wektory z faces_w przesunięte w kierunku uśrednionej twarzy"""
        w_avg = self.Gs.get_var('dlatent_avg')
        return w_avg + (faces_w - w_avg) * self.truncation

    def __set_synthesis_kwargs(self,minibatch_size = 3):
        """Za pierwszym razem tworzy keyword arguments do gnereowania,
        następnie może być użyta do zienienia minibatch_size"""
        if len(self.synthesis_kwargs)==0:
            Gs_syn_kwargs = dnnlib.EasyDict()
            Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                                  nchw_to_nhwc=True)
            Gs_syn_kwargs.randomize_noise = False
            self.synthesis_kwargs = Gs_syn_kwargs

        Gs_syn_kwargs.minibatch_size = minibatch_size