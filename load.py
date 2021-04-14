import pretrained_networks

def load(network_pkl_path):
    return pretrained_networks.load_networks(network_pkl_path)