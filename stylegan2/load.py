
def load(network_pkl_path):
    import os
    print(os.getcwd())
    return pretrained_networks.load_networks(network_pkl_path)