from stylegan2.generator import generator
def main():
    main_generator = generator(network_pkl_path="/stylegan2/stylegan2-ffhq-config-f.pkl",
                               direction_path="stylegan2/stylegan2directions/dominance.npy", coefficient=1.0,
                               truncation=0.7, n_levels=3, n_photos=10, type_of_preview="manipulation",
                               result_dir="/results")
    plt.imshow(generator.__generate_preview_face_manip())


main()