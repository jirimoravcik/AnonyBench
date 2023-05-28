import os, logging, pandas as pd, math
from deepface import DeepFace
from tqdm import tqdm
from contextlib import contextmanager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..cli import Configuration
# Possibly try detector_backend='skip' and see how it impacts the result

LOG = logging.getLogger('Face reidentification')
DEFAULT_THRESHOLD = 0.65 # ~LFW with FP of 0.005

@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)

def compute_threshold(config: 'Configuration'):
    """Compute threshold for face reidentification based on false positivity rate."""
    LOG.info(f'Computing threshold for face reidentification')
    distances = []
    lines = []
    with open(config.non_matching_pairs_filepath) as f:
        lines.extend(filter(lambda x: x != '', f.readlines()))
    
    for i in tqdm(range(0, len(lines), config.batch_size), desc='Threshold'):
        with all_logging_disabled():
            verify_input = []
            for line in lines[i:i + config.batch_size]:
                img1, img2 = line.strip().split(',')
                verify_input.append([os.path.join(config.original_dataset_dir, img1), os.path.join(config.original_dataset_dir, img2)])
            for img1, img2 in verify_input:
                verify_result = DeepFace.verify(img1_path=img1, img2_path=img2, detector_backend=config.detector, enforce_detection=False, model_name='ArcFace')
                distances.append(verify_result['distance'])
    
    fp_rate = len(distances) * config.fp_rate
    low_idx, high_idx = math.floor(fp_rate), math.ceil(fp_rate)
    sorted_distances = sorted(distances)
    return (sorted_distances[low_idx] + sorted_distances[high_idx]) / 2

def run_reidentify_faces_benchmark(config: 'Configuration'):
    """Run the face reidentification benchmark."""
    threshold = compute_threshold(config) if config.non_matching_pairs_filepath else DEFAULT_THRESHOLD
    LOG.info(f'Running face reidentification with distance threshold {threshold}')
    rel_file_path_with_pairs = []
    for dirpath, _, fnames in os.walk(config.original_dataset_dir):
        for fname in fnames:
            source_file_path = os.path.join(dirpath, fname)
            rel_file_path = os.path.relpath(source_file_path, start=config.original_dataset_dir)
            rel_file_path_with_pairs.append((rel_file_path, source_file_path, os.path.join(config.anonymized_dataset_dir, rel_file_path)))
    data = {'file_path': [], 'verified': [], 'distance': [], 'threshold': [], 'fp_rate': [] }
    for i in tqdm(range(0, len(rel_file_path_with_pairs), config.batch_size), desc='Reidentification'):
        with all_logging_disabled():
            rel_file_paths = list(map(lambda x: x[0], rel_file_path_with_pairs[i:i + config.batch_size]))
            orig_file_paths = list(map(lambda x: x[1], rel_file_path_with_pairs[i:i + config.batch_size]))
            anon_file_paths = list(map(lambda x: x[2], rel_file_path_with_pairs[i:i + config.batch_size]))
            for orig, anon, rel_file_path in zip(orig_file_paths, anon_file_paths, rel_file_paths):
                verify_result = DeepFace.verify(img1_path=orig, img2_path=anon, detector_backend=config.detector, enforce_detection=False, model_name='ArcFace')
                data['file_path'].append(rel_file_path)
                actual_distance = verify_result['distance']
                data['distance'].append(actual_distance)
                data['verified'].append(actual_distance < threshold)
                data['threshold'].append(threshold)
                data['fp_rate'].append(config.fp_rate)

    output_path = os.path.join(config.output_dir, f'face_reidentification__{os.path.basename(os.path.normpath(config.anonymized_dataset_dir))}.csv')
    LOG.info(f'Saving results to {output_path}')
    pd.DataFrame.from_dict(data).to_csv(output_path)
