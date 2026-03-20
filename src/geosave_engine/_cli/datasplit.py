import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse 
import re

SEED = 10
IMAGE_EXTENSIONS = ['*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.jp2']

def extract_id(filename):
    stem = Path(filename).stem
    match = re.findall(r'\d+', stem)
    
    if match:
        return match[-1]  # Return last numeric sequence
    
    return stem  # Fallback to full stem if no numbers found

def seg_split(img_dir, mask_dir, out_dir, train_val_test):

    assert sum(train_val_test) == 1.0, "Train/val/test splits must sum to 1.0"
    
    # Collect all images
    images, labels = [], []
    for ext in IMAGE_EXTENSIONS:
        images.extend(Path(img_dir).glob(ext))
    # Collect all masks
    for ext in IMAGE_EXTENSIONS:
        labels.extend(Path(mask_dir).glob(ext))

    images = sorted(images)
    labels = sorted(labels)
    
    label_dict = {extract_id(label.name): label for label in labels}
    
    # Match images to masks by ID
    labeled_pairs = []
    unlabeled_imgs = []
    
    for img in images:
        img_name = img.name
        img_id = extract_id(img_name)
        if img_id in label_dict:
            label_name = label_dict[img_id].name
            labeled_pairs.append((img_name, label_name))
        else:
            unlabeled_imgs.append(img_name)
    
    if not labeled_pairs:
        raise ValueError(f"No matching image-mask pairs found. Check that filenames match (ignoring extensions)")
    
    print(f"Found {len(labeled_pairs)} labeled images, {len(unlabeled_imgs)} unlabeled images")
    
    X_labeled = [pair[0] for pair in labeled_pairs]
    y_labeled = [pair[1] for pair in labeled_pairs]
    
    train_split = train_val_test[0]
    val_split = train_val_test[1] / (train_val_test[1] + train_val_test[2])

    X_train, X_rem, y_train, y_rem = train_test_split(X_labeled, y_labeled, train_size=train_split, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, train_size=val_split, random_state=SEED)

    train_df = pd.DataFrame({'Image': X_train, 'Label': y_train})
    val_df = pd.DataFrame({'Image': X_val, 'Label': y_val})
    test_df = pd.DataFrame({'Image': X_test, 'Label': y_test})

    os.makedirs(out_dir, exist_ok=True)

    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    if unlabeled_imgs:
        unlabeled_df = pd.DataFrame({'Image': unlabeled_imgs, 'Label': [None]*len(unlabeled_imgs)})
        unlabeled_df.to_csv(os.path.join(out_dir, 'unlabeled.csv'), index=False)
    
    print(f'Split saved to {out_dir}')


def main():
    parser = argparse.ArgumentParser(description='Segmented Dataset Splitter')
    parser.add_argument('--root', type=str, required=True, help='Directory containing images')
    parser.add_argument('--out_dir', default=None, help='Output directory for splits')
    parser.add_argument('--train_val_test', type=float, nargs=3, default=[0.7, 0.15, 0.15], help='Train/validation/test split ratios')

    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(args.root, 'splits')

    img_dir = os.path.join(args.root, 'images')
    mask_dir = os.path.join(args.root, 'labels')

    seg_split(img_dir, mask_dir, args.out_dir, args.train_val_test)


if __name__ == '__main__':
    main()