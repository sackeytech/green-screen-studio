import cv2
import numpy as np
import os

def evaluate_five_frames(folder="eval_frames"):
    accuracies = []
    print("-" * 55)
    print(f"{'Frame':<10} | {'Correct Pixels':<18} | {'Accuracy (%)':<15}")
    print("-" * 55)

    for i in range(1, 6):
        algo_path = os.path.join(folder, f"algo_mask_{i}.png")
        gt_path = os.path.join(folder, f"gt_mask_{i}.png")

        if not (os.path.exists(algo_path) and os.path.exists(gt_path)):
            print(f"Frame {i:<4} | Missing file(s)! Need both '{algo_path}' and '{gt_path}'")
            continue

        # Load both masks in grayscale
        algo_mask = cv2.imread(algo_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Ensure both are perfectly binary (0 or 255) for an exact pixel-by-pixel comparison
        algo_bin = np.where(algo_mask >= 128, 255, 0)
        gt_bin = np.where(gt_mask >= 128, 255, 0)

        # Count how many pixels perfectly match
        correct_pixels = np.sum(algo_bin == gt_bin)
        total_pixels = gt_bin.size
        
        # Calculate percentage
        pct_correct = (correct_pixels / total_pixels) * 100.0
        accuracies.append(pct_correct)

        print(f"Frame {i:<4} | {correct_pixels:>8}/{total_pixels:<9} | {pct_correct:>10.2f}%")

    if accuracies:
        print("-" * 55)
        print(f"Mean Accuracy Across {len(accuracies)} Frames: {np.mean(accuracies):.2f}%")
        print("-" * 55)

if __name__ == "__main__":
    evaluate_five_frames()