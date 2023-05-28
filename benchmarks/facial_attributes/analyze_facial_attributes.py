import logging, os, pandas as pd
from contextlib import contextmanager
from tqdm import tqdm
from deepface import DeepFace
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..cli import Configuration
LOG = logging.getLogger('Facial attributes')
WANTED_ACTIONS = ('age', 'gender', 'race')
WANTED_ATTRIBUTES = ['dominant_gender', 'age', 'dominant_race']

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

# DeepFace returns a dict of dicts indexed like 'instance_1', 'instance_2', etc.
def deepface_dict_to_list(deepface_dict, batch_size):
    ret = []
    for i in range(1, batch_size + 1):
        if deepface_dict.get(f'instance_{i}') is None:
            break
        ret.append(deepface_dict[f'instance_{i}'])
    return ret

def run_analyze_facial_attributes_benchmark(config: 'Configuration'):
    """Run the facial attributes benchmark."""
    rel_file_path_with_pairs = []
    for dirpath, _, fnames in os.walk(config.original_dataset_dir):
        for fname in fnames:
            source_file_path = os.path.join(dirpath, fname)
            rel_file_path = os.path.relpath(source_file_path, start=config.original_dataset_dir)
            rel_file_path_with_pairs.append((rel_file_path, source_file_path, os.path.join(config.anonymized_dataset_dir, rel_file_path)))
    
    data = {'file_path': []}
    for i in tqdm(range(0, len(rel_file_path_with_pairs), config.batch_size)):
        rel_file_paths = list(map(lambda x: x[0], rel_file_path_with_pairs[i:i + config.batch_size]))
        orig_file_paths = list(map(lambda x: x[1], rel_file_path_with_pairs[i:i + config.batch_size]))
        anon_file_paths = list(map(lambda x: x[2], rel_file_path_with_pairs[i:i + config.batch_size]))
        with all_logging_disabled():
            for rel_file_path, orig_file_path, anon_file_path in zip(rel_file_paths, orig_file_paths, anon_file_paths):
                orig = DeepFace.analyze(orig_file_path, detector_backend=config.detector, enforce_detection=False, actions=WANTED_ACTIONS, silent=True)[0]
                anon = DeepFace.analyze(anon_file_path, detector_backend=config.detector, enforce_detection=False, actions=WANTED_ACTIONS, silent=True)[0]
                data['file_path'].append(rel_file_path)
                for attr in WANTED_ATTRIBUTES:
                    if f'orig_{attr}' not in data:
                        data[f'orig_{attr}'] = []
                        data[f'anon_{attr}'] = []
                    data[f'orig_{attr}'].append(orig[attr])
                    data[f'anon_{attr}'].append(anon[attr])

    output_path = os.path.join(config.output_dir, f'facial_attributes__{os.path.basename(os.path.normpath(config.anonymized_dataset_dir))}.csv')
    LOG.info(f'Saving results to {output_path}')
    pd.DataFrame.from_dict(data).to_csv(output_path)    
