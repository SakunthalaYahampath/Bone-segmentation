from args import get_args
import os
import pandas as pd
import torch
from dataset import Knee_Dataset
from torch.utils.data import DataLoader
from model import UNetLext
from trainer import train_model


def main():
    args = get_args()

    train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
    val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

    train_dataset = Knee_Dataset(train_set)
    val_dataset = Knee_Dataset(val_set)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    #initializing the model
    model=UNetLext(input_channels=1,
                   output_channels=1,
                   pretrained=False,
                   path_pretrained="",
                   restore_weights=False,
                   path_weights=""


    )
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, val_loader, device)

    os.makedirs('session',exist_ok=True)
    model_save_path = 'session/model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved at {model_save_path}')

if __name__ == '__main__':
    main()