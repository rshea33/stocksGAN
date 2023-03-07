
dict_main = {
        'train_size': [0.7, 0.8, 0.9, 0.95],
        'epochs': [25, 50, 100, 200, 250, 300, 400, 500],
        'batch_size': [32, 64, 128, 256, 512, 1024],
        'lr': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'b1': [0.5, 0.6, 0.7, 0.8, 0.9],
        'b2': [0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999],
        'clip_value': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, None],
        'random_state': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

default_params = {
        'train_size': 0.7,
        'epochs': 100,
        'batch_size': 64,
        'lr': 0.0001,
        'b1': 0.9,
        'b2': 0.999,
        'clip_value': None,
        'random_state': 0
    }