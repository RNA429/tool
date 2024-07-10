import os
import sys
import shutil
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm
import cv2
from multiprocessing import Pool

def apply_window_level(image_data, level=None, window=None):
    if level is not None and window is not None:
        min_val = level - (window / 2)
        max_val = level + (window / 2)
        clipped_data = np.clip(image_data, min_val, max_val)
        return clipped_data
    return image_data

def split_dataset(img_dir, mask_dir, out_dir, train_ratio=0.8):
    # Create output directories
    img_train = os.path.join(out_dir, 'images', 'train')
    img_val = os.path.join(out_dir, 'images', 'val')
    label_train = os.path.join(out_dir, 'labels', 'train')
    label_val = os.path.join(out_dir, 'labels', 'val')

    os.makedirs(img_train, exist_ok=True)
    os.makedirs(img_val, exist_ok=True)
    os.makedirs(label_train, exist_ok=True)
    os.makedirs(label_val, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith('nii.gz')]
    
    random.seed(6)
    random.shuffle(img_files)

    split_index = int(len(img_files) * train_ratio)
    classification = os.path.basename(img_dir).lower()
    train_files = img_files[:split_index]
    val_files = img_files[split_index:]

    for file in train_files:
        shutil.copy(os.path.join(img_dir, file), os.path.join(img_train, f'{classification}_'+file))
        shutil.copy(os.path.join(mask_dir, file), os.path.join(label_train, f'{classification}_'+file))
        
    for file in val_files:
        shutil.copy(os.path.join(img_dir, file), os.path.join(img_val, f'{classification}_'+file))
        shutil.copy(os.path.join(mask_dir, file), os.path.join(label_val, f'{classification}_'+file))
        
    print(f"Training images: {len(train_files)}, Validation images: {len(val_files)}")
    
    return img_train, img_val, label_train, label_val, classification

def nii_to_png_masks(args, level=50, window=350):
    nii_file, output_dir  = args
    img = nib.load(nii_file)
    img_data = img.get_fdata()
    
    base_name = os.path.basename(nii_file).split('.')[0]
    
    if 'mask' not in nii_file and 'labels' not in nii_file:
        img_data = apply_window_level(img_data, level, window)

    os.makedirs(output_dir, exist_ok=True)
    for i in range(img_data.shape[2]):
        slice_data = img_data[:, :, i]
        if 'mask' in nii_file or 'labels' in nii_file:
            slice_data[slice_data==1] = 2
            slice_data[slice_data==3] = 2
            # Create two masks
            mask1 = (slice_data == 2)
            mask2 = (slice_data == 4)

            # Create a blank grayscale image
            combined_mask = np.zeros_like(slice_data, dtype=np.uint8)

            # Assign grayscale values based on the masks
            combined_mask[mask1] = 1  # Grayscale value for mask1
            if 'spt' in nii_file.lower():
                combined_mask[mask2] = 2
            elif 'mcn' in nii_file.lower():
                combined_mask[mask2] = 3
            elif 'scn' in nii_file.lower():
                combined_mask[mask2] = 4
            elif 'ipmn' in nii_file.lower():
                combined_mask[mask2] = 5
            # Save the combined mask

            # Assuming combined_mask is your numpy array representing the grayscale image
            combined_mask = Image.fromarray(combined_mask)
            combined_mask.save(os.path.join(output_dir, f"{base_name}_slice_{i:03d}.png"))
            # plt.imshow(combined_mask, cmap='gray')  # Display as grayscale, set min and max values
            # plt.imshow(combined_mask, cmap='gray', vmin=0, vmax=255)  # Display as grayscale, set min and max values
        else:
            plt.imshow(slice_data, cmap='gray')  # Display as grayscale
            plt.axis('off')  # Hide the axes
            output_path = os.path.join(output_dir, f"{base_name}_slice_{i:03d}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

def is_mask_mostly_black(mask_path, threshold=0.999):
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    # mask = mask.point(lambda p: p > 40 and 255)  # Threshold at 40 (0-255 scale)
    mask_array = np.array(mask)
    total_pixels = mask_array.size
    black_pixels = np.sum(mask_array == 0)
    black_ratio = black_pixels / total_pixels
    return black_ratio > threshold

def delete_files(img_path, mask_path):
    try:
        os.remove(img_path)
        os.remove(mask_path)
        print(f"Deleted: {img_path} and {mask_path}")
    except Exception as e:
        print(f"Error deleting files {img_path} and {mask_path}: {e}")

def process_0_files(image_path, mask_path, classification):
    # images = sorted(os.listdir(image_path))
    masks = sorted(os.listdir(mask_path))
    # image_list = [os.path.join(image_path, each) for each in images if each.endswith('.png')]
    # mask_list = [os.path.join(mask_path, each) for each in masks if (each.endswith('.png'))]
    mask_list = [os.path.join(mask_path, each) for each in masks if (each.endswith('.png') and classification in each)]
    # image_list = glob.glob(image_path+'/*.png')
    # mask_list = glob.glob(mask_path+'/*.png')
    for mask_path in  mask_list:
        if is_mask_mostly_black(mask_path):
            if os.path.exists(os.path.join(image_path, os.path.basename(mask_path))):
                delete_files(os.path.join(image_path, os.path.basename(mask_path)), mask_path)
            else:
                os.remove(mask_path)
            
def mask_to_yolo_multiclass(mask_image_path, output_txt_path, class_mapping, anormal, min_area=100):
    # 读取掩码图像
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape

    with open(output_txt_path, 'w') as f:
        for class_value, class_id in class_mapping.items():
            # 创建掩码，只保留当前类别的像素
            class_mask = (mask == class_value).astype(np.uint8)
            total_num = np.sum(mask == class_value)  
            # 找到所有的轮廓
            if total_num > min_area:
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 合并所有轮廓
                if contours:
                # for each_count in contours:
                    all_contours = np.vstack(contours)
                    x, y, w, h = cv2.boundingRect(all_contours)

                    # 计算YOLO格式的中心坐标和宽高（相对于图像尺寸）
                    area = w * h
                    if area < min_area:
                        continue  # 跳过面积小于阈值的轮廓

                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    width_ratio = w / width
                    height_ratio = h / height

                    # 写入YOLO格式的标注
                    f.write(f"{class_id} {x_center} {y_center} {width_ratio} {height_ratio}\n")
                

def mask_to_roi(mask_path, class_mapping, anormal, min_area=100):
    # 读取掩码图像
    image_path = mask_path.replace('labels', 'images')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    for class_value, class_id in class_mapping.items():
        # 创建掩码，只保留当前类别的像素
        class_mask = (mask == class_value).astype(np.uint8)
        num255 = np.sum(mask == anormal)
        if num255 >= min_area:
            class_mask = ((mask == anormal) | (mask == class_value)).astype(np.uint8)
        else:
            os.remove(image_path)
            os.remove(mask_path)
            break
        
        # 找到所有的轮廓
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 合并所有轮廓
        if contours:
            all_contours = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_contours)

            # 计算YOLO格式的中心坐标和宽高（相对于图像尺寸）
            area = w * h
            if area < min_area:
                continue  # 跳过面积小于阈值的轮廓
            
            annotated_region = image[y-5:y+h+5, x-5:x+w+5]
            annotated_mask = mask[y-5:y+h+5, x-5:x+w+5]

            # Save the extracted region as an image
            annotated_mask = Image.fromarray(annotated_mask)
            annotated_mask.save(mask_path)
            cv2.imwrite(image_path, annotated_region)
             
if __name__ == "__main__":
    img_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    out_dir = sys.argv[3]

    print('Split train and val...')
    img_train, img_val, label_train, label_val, classification = split_dataset(img_dir, mask_dir, out_dir, train_ratio=0.8)

    print('Processing nii.gz files...')
    files_to_process = []

    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if file.endswith('nii.gz'):
                nii_file = os.path.join(root, file)
                files_to_process.append((nii_file, root))

    with Pool(20) as pool:
        list(tqdm(pool.imap(nii_to_png_masks, files_to_process), total=len(files_to_process), desc='Processing nii.gz'))


    print('Deleting all black images and masks...')
    process_0_files(img_train, label_train, classification)
    process_0_files(img_val, label_val, classification)

   
    anormal = None
    class_mapping = {1: 0, 2: 1, 3: 1, 4: 1, 5: 1}
    min_area = 100
    class_id = 0
    for root, dirs, files in tqdm(os.walk(out_dir), desc='convert to txt'):
        for file in files:
            if file.endswith('png') and 'labels' in root and classification in file:
                label_file = os.path.join(root,file)
                if 'spt' in label_file.lower():
                    anormal = 2  # 掩码2的灰度值
                elif 'mcn' in label_file.lower():
                    anormal = 3
                elif 'scn' in label_file.lower():
                    anormal = 4
                elif 'ipmn' in label_file.lower():
                    anormal = 5
                # mask_to_roi(label_file, class_mapping, anormal, min_area)
                out_path = label_file.replace('.png','.txt')
                mask_to_yolo_multiclass(label_file, out_path, class_mapping, anormal, min_area)
                os.remove(label_file)
            if file.endswith('nii.gz'):
                file_path = os.path.join(root,file)
                os.remove(file_path)
