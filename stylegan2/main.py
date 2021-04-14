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
import pretrained_networks, dnnlib

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
        """Przełączniki co wywołać w zależności od wartości type_of_preview"""
        pass

    def change_face(self):
        all_z = np.random.randn(1, *Gs.input_shape[1:])
        all_w = self.__map_vectors(all_z)
        all_w = self.__truncate_vectors(all_w)

    def generate(self):
        """Zapisuje wyniki, na razie n_levels=1 """
        result_dir = Path(dnnlib.submit_config.run_dir_root)
        output_dir = Path(output_dir)

        if direction_path is not None:
            direction = np.load(direction_path)

        images_dir = result_dir / 'images'
        dlatents_dir = result_dir / 'dlatents'
        output_tsv = output_dir / 'out.tsv'

        images_dir.mkdir(exist_ok=True, parents=True)
        dlatents_dir.mkdir(exist_ok=True, parents=True)
        output_dir.mkdir(exist_ok=True)

        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
        w_avg = Gs.get_var('dlatent_avg')

        Gs_syn_kwargs = dnnlib.EasyDict()
        Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                              nchw_to_nhwc=True)
        Gs_syn_kwargs.randomize_noise = False
        Gs_syn_kwargs.minibatch_size = minibatch_size

        latents_from = 0
        latents_to = 8

        for i in tqdm(range(num // minibatch_size)):
            all_z = np.random.randn(minibatch_size, *Gs.input_shape[1:])
            all_w = Gs.components.mapping.run(all_z, None)
            all_w = w_avg + (all_w - w_avg) * truncation_psi

            if direction_path is not None:
                assert coeff is not None
                pos_w = all_w.copy()
                neg_w = all_w.copy()

                for j in range(len(all_w)):
                    pos_w[j][latents_from:latents_to] = \
                        (pos_w[j] + coeff * direction)[latents_from:latents_to]
                    neg_w[j][latents_from:latents_to] = \
                        (neg_w[j] - coeff * direction)[latents_from:latents_to]

                pos_images = Gs.components.synthesis.run(pos_w,
                                                         **Gs_syn_kwargs)
                neg_images = Gs.components.synthesis.run(neg_w,
                                                         **Gs_syn_kwargs)

                for j in range(len(all_w)):
                    pos_image_pil = PIL.Image.fromarray(pos_images[j], 'RGB')
                    pos_image_pil.save(
                        images_dir / 'tr_{}_{}.png'.format(i * minibatch_size +
                                                           j, coeff))

                    neg_image_pil = PIL.Image.fromarray(neg_images[j], 'RGB')
                    neg_image_pil.save(
                        images_dir / 'tr_{}_-{}.png'.format(i * minibatch_size +
                                                            j, coeff))

            all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs)

            for j, (dlatent, image) in enumerate(zip(all_w, all_images)):
                image_pil = PIL.Image.fromarray(image, 'RGB')
                image_pil.save(images_dir / (str(i * minibatch_size + j) + '.png'))
                np.save(dlatents_dir / (str(i * minibatch_size + j) + '.npy'),
                        dlatent[0])

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

    def __map_vectors(self, faces_z):
         """Przyjmuje array wektorów z koordynatami twarzy w Z-space, gdzie losowane są wektory,
         zwraca array przerzucony do w-space, gdzie dzieje się manipulacja"""
         return self.Gs.components.mapping.run(faces_z, None)

    def __truncate_vectors(self, faces_w):
        """Zwraca wektory z faces_w przesunięte w kierunku uśrednionej twarzy"""
        w_avg = self.Gs.get_var('dlatent_avg')
        return w_avg + (faces_w - w_avg) * self.truncation

    def __kwargs(self):  # Adam
        Gs_syn_kwargs = dnnlib.EasyDict()
        Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                              nchw_to_nhwc=True)
        Gs_syn_kwargs.randomize_noise = False
        Gs_syn_kwargs.minibatch_size = 1
        self.synthesis_kwargs = Gs_syn_kwargs



def main():
    main_generator = generator(network_pkl_path="stylegan2-ffhq-config-f.pkl",direction_path="stylegan2/stylegan2directions/dominance.npy",coefficient=1.0,truncation=0.7,n_levels=3,n_photos=10,type_of_preview="manipulation",result_dir="/results")
    plt.imshow(generator.__generate_preview_face_manip())


main()