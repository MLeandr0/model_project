from model import train_and_save_model
from data_colection import download_data

if __name__ == "__main__":
    download_data()
    train_and_save_model()