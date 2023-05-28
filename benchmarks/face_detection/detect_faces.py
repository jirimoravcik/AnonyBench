
import cv2, numpy as np, pandas as pd, tempfile
import os, logging
from tqdm import tqdm
from deepface import DeepFace
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..cli import Configuration

LOG = logging.getLogger('Face detection')
BASELINE_DATASETS_DIRS = ['boxed', 'blurred', 'pixelized', 'blurred-full']
CACHE_DIR_BASE = os.path.join(tempfile.gettempdir(), 'face-anonymization-benchmark')

def pixelize(image, downscale_by):
    """Downscale image and then upscale back to original size."""
    height, width = image.shape[:2]
    downscaled = cv2.resize(image, (width // downscale_by, height // downscale_by), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_NEAREST)

def get_source_file_paths(original_dataset_dir):
    source_file_paths = []
    for dirpath, _, fnames in os.walk(original_dataset_dir):
        for fname in fnames:
            source_file_paths.append(os.path.join(dirpath, fname))
    return source_file_paths

def prepare_baseline_face_detection_datasets(config: 'Configuration'):
    """Prepares datasets with blurred, pixelized and boxed faces."""
    cache_dir = os.path.join(CACHE_DIR_BASE, os.path.basename(config.original_dataset_dir))
    if os.path.exists(cache_dir):
        return
    for dir in BASELINE_DATASETS_DIRS:
        dir_path = os.path.join(cache_dir, dir)
        os.makedirs(dir_path, exist_ok=True)

    source_file_paths = get_source_file_paths(config.original_dataset_dir)
    for source_file_path in tqdm(source_file_paths, desc='Preparation'):
        dest_file_rel_path = os.path.relpath(source_file_path, start=config.original_dataset_dir)
        for dir in BASELINE_DATASETS_DIRS:
            dir_path = os.path.join(cache_dir, dir, os.path.split(dest_file_rel_path)[0])
            os.makedirs(dir_path, exist_ok=True)
        image = cv2.imread(source_file_path)
        rects = []
        faces = DeepFace.extract_faces(img_path=image, detector_backend=config.detector, enforce_detection=False, align=False)
        faces = [f for f in faces if f['confidence'] > 0]
        for f in faces:
            x, y, w, h = f['facial_area']['x'], f['facial_area']['y'], f['facial_area']['w'], f['facial_area']['h']
            rects.append((x, y, x + w, y + h))
        for rect in rects:
            # mask
            mask = np.zeros(image.shape, dtype=np.uint8)
            width_c = (rect[2] - rect[0]) // 10
            heigth_c = (rect[3] - rect[1]) // 10
            rescaled_left = 0 if rect[0] - width_c < 0 else rect[0] - width_c
            rescaled_top = 0 if rect[1] - heigth_c < 0 else rect[1] - heigth_c
            rescaled_right = image.shape[1] if rect[2] + width_c > image.shape[1] else rect[2] + width_c
            rescaled_bottom = image.shape[0] if rect[3] + heigth_c > image.shape[0] else rect[3] + heigth_c
            cv2.rectangle(mask, (rescaled_left, rescaled_top), (rescaled_right, rescaled_bottom), (255, 255, 255), -1)

            boxed = image.copy()
            cv2.rectangle(boxed, (rescaled_left, rescaled_top), (rescaled_right, rescaled_bottom), (0, 0, 0), -1)
            cv2.imwrite(os.path.join(cache_dir, 'boxed', dest_file_rel_path), boxed)
            blurred = cv2.blur(image, (32, 32))
            cv2.imwrite(os.path.join(cache_dir, 'blurred', dest_file_rel_path), np.where(mask == (255, 255, 255), blurred, image))
            pixelized = pixelize(image, 16)
            cv2.imwrite(os.path.join(cache_dir, 'pixelized', dest_file_rel_path), np.where(mask == (255, 255, 255), pixelized, image))

        blurred = cv2.blur(image, (32, 32))

        cv2.imwrite(os.path.join(cache_dir, 'blurred-full', dest_file_rel_path), blurred)

def detect_faces(config: 'Configuration'):
    """Detect faces for all given datasets."""
    cache_dir = os.path.join(CACHE_DIR_BASE, os.path.basename(config.original_dataset_dir))
    other_folders = [(config.anonymized_dataset_dir, 'anonymized'), *[(os.path.join(cache_dir, dir), dir) for dir in BASELINE_DATASETS_DIRS]]
    data = {'file_path': []}
    source_file_paths = get_source_file_paths(config.original_dataset_dir)
    for source_file_path in tqdm(source_file_paths, desc='Detection'):
        rel_file_path = os.path.relpath(source_file_path, start=config.original_dataset_dir)
        data['file_path'].append(rel_file_path)
        for file_path, identifier in [(source_file_path, 'original'), *[(os.path.join(d, rel_file_path), id) for d, id in other_folders]]:
            if identifier not in data:
                data[identifier] = []
            if not os.path.exists(file_path):
                data[identifier].append('DOES_NOT_EXIST')
                continue
            image = cv2.imread(file_path)
            # detect faces
            faces = DeepFace.extract_faces(img_path=image, detector_backend=config.detector, enforce_detection=False, align=False)
            faces = [f for f in faces if f['confidence'] > 0]
            face_found = len(faces) > 0
            data[identifier].append('FOUND' if face_found else 'NOT_FOUND')

    output_path = os.path.join(config.output_dir, f'face_detection__{os.path.basename(os.path.normpath(config.anonymized_dataset_dir))}.csv')
    LOG.info(f'Saving results to {output_path}')
    pd.DataFrame.from_dict(data).to_csv(output_path)

def run_face_detection_benchmark(config: 'Configuration'):
    """Run the face detection benchmark."""
    prepare_baseline_face_detection_datasets(config)
    detect_faces(config)
