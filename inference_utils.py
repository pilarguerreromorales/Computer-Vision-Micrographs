import torch
import numpy as np
import tqdm
def sliding_window_inference(model, image, window_size, overlap=0.5):
    """Perform sliding window inference on large images"""
    model.eval()

    # Get dimensions
    _, height, width = image.shape
    stride = int(window_size * (1 - overlap))

    # Calculate number of windows needed
    n_h = int(np.ceil((height - window_size) / stride) + 1)
    n_w = int(np.ceil((width - window_size) / stride) + 1)

    # Create empty prediction map and count map for averaging
    pred_map = torch.zeros((1, height, width)).to(model.device)
    count_map = torch.zeros((1, height, width)).to(model.device)

    # Slide window over image
    with torch.no_grad():
        for i in range(n_h):
            for j in range(n_w):
                # Calculate window boundaries
                h_start = min(i * stride, height - window_size)
                w_start = min(j * stride, width - window_size)
                h_end = h_start + window_size
                w_end = w_start + window_size

                # Extract window
                window = image[:, h_start:h_end, w_start:w_end]

                # If window is smaller than window_size, pad it
                if window.shape[1:] != (window_size, window_size):
                    pad_h = window_size - window.shape[1]
                    pad_w = window_size - window.shape[2]
                    window = torch.nn.functional.pad(window, (0, pad_w, 0, pad_h))

                # Make prediction
                window = window.unsqueeze(0)  # Add batch dimension
                pred = model(window)
                pred = pred.squeeze(0)  # Remove batch dimension

                # If window was padded, remove padding from prediction
                if window.shape[2] - h_end + h_start > 0 or window.shape[3] - w_end + w_start > 0:
                    pred = pred[:, :h_end - h_start, :w_end - w_start]

                # Add prediction to map
                pred_map[:, h_start:h_end, w_start:w_end] += pred
                count_map[:, h_start:h_end, w_start:w_end] += 1

    # Average overlapping predictions
    final_pred = pred_map / count_map
    return final_pred.cpu()