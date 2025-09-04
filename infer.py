#!/usr/bin/env python3
import os
import pandas as pd
import torch
from model import MicrographCleaner
from dataset import InferenceMicrographDataset, decode_array
from inference_utils import sliding_window_inference
import matplotlib.pyplot as plt
import tqdm


def main():
    # Create predictions directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)

    # Parameters
    WINDOW_SIZE = 512
    THRESHOLD = 0.5
    OVERLAP = 0.5

    # Load model
    model = MicrographCleaner.load_from_checkpoint('final_checkpoint.ckpt', map_location='cpu')
    model.eval()

    # Load test data
    test_df = pd.read_csv('test.csv')
    test_dataset = InferenceMicrographDataset(test_df, window_size=WINDOW_SIZE)

    # Process each image
    unique_ids = set()
    model.eval()
    with torch.inference_mode():
        for idx in tqdm.tqdm(range(len(test_dataset))):
            image, image_id, (pad_h, pad_w) = test_dataset[idx]

            # Skip if already processed
            if image_id in unique_ids:
                continue
            unique_ids.add(image_id)

            # Perform inference
            pred = sliding_window_inference(
                model,
                image,
                window_size=WINDOW_SIZE,
                overlap=OVERLAP
            )

            # Remove padding if necessary
            if pad_h > 0:
                pred = pred[..., :-pad_h, :]
            if pad_w > 0:
                pred = pred[..., :-pad_w]

            # Convert to binary mask
            pred_mask = (pred > THRESHOLD).cpu().numpy()[0]

            # Create visualization
            orig_image = decode_array(test_df.iloc[idx]['image'])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(orig_image, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')

            ax2.imshow(pred_mask, cmap='gray')
            ax2.set_title('Predicted Mask')
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(f'predictions/{image_id}_prediction.png')
            plt.close()


if __name__ == "__main__":
    main()