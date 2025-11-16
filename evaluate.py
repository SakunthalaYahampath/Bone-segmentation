import os
import pandas as pd
import torch
import cv2

from args import get_args
from dataset import Knee_Dataset
from torch.utils.data import DataLoader
from model import UNetLext


def evaluate():
    args = get_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_csv_path = os.path.join(args.csv_dir, "test.csv")
    test_df = pd.read_csv(test_csv_path)

    test_dataset = Knee_Dataset(test_df, has_mask=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_path = os.path.join(args.out_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = UNetLext(
        input_channels=1,
        output_channels=1,
        pretrained=False,
        path_pretrained="",
        restore_weights=False,
        path_weights=""
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    pred_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device=device)
            name = batch["name"][0]

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            pred_np = preds[0, 0].cpu().numpy() * 255.0
            pred_np = pred_np.astype("uint8")

            base = os.path.splitext(name)[0]
            out_path = os.path.join(pred_dir, base + "_pred.png")

            cv2.imwrite(out_path, pred_np)
            print(f"Saved prediction: {out_path}")


if __name__ == "__main__":
    evaluate()
