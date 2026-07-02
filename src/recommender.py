from surprise import SVD, KNNBasic, NormalPredictor
from surprise.model_selection import train_test_split

def create_svd_model(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
    return SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, verbose=True)

def create_knn_user_model(k=40, sim_option='cosine'):
    sim_options = {'name': sim_option, 'user_based': True}
    return KNNBasic(k=k, sim_options=sim_options, verbose=True)

def create_knn_item_model(k=40, sim_option='cosine'):
    sim_options = {'name': sim_option, 'user_based': False}
    return KNNBasic(k=k, sim_options=sim_options, verbose=True)

def create_random_model():
    return NormalPredictor()

def train_model(model, data):
    trainset = data.build_full_trainset()
    model.fit(trainset)
    return model