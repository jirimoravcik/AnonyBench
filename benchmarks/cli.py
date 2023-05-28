import os, argparse, logging, filetype
from PIL import Image
from face_detection.detect_faces import run_face_detection_benchmark
from face_reidentification.reidentify_faces import run_reidentify_faces_benchmark
from facial_attributes.analyze_facial_attributes import run_analyze_facial_attributes_benchmark
from gan_metrics.compute_gan_metrics import run_compute_gan_metrics_benchmark
from classifier.classify import run_classify_benchmark

POSSIBLE_BENCHMARKS = ['face_detection', 'face_reidentification', 'facial_attributes', 'gan_metrics', 'classifier']
LOG = logging.getLogger('CLI')

class Configuration:
    """Used to pass CLI configuration to individual benchmark suites."""
    def __init__(self,
        *,
        original_dataset_dir,
        anonymized_dataset_dir,
        non_matching_pairs_filepath,
        use_gpu,
        batch_size,
        fp_rate,
        output_dir,
        detector,
    ) -> None:
        self.original_dataset_dir = original_dataset_dir
        self.anonymized_dataset_dir = anonymized_dataset_dir
        self.non_matching_pairs_filepath = non_matching_pairs_filepath
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.fp_rate = fp_rate
        self.output_dir = output_dir
        self.detector = detector


def dir_path(string):
    """Check if path is an existing directory."""
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def create_dir_if_non_existent(string):
    """Check if dir exists. If not, attempt to create it."""
    if os.path.isdir(string):
        return string
    elif not os.path.exists(string):
        os.makedirs(string)
        return string

def file_exists(string):
    """Check if file exists."""
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)

parser = argparse.ArgumentParser(
    prog = 'AnonyBench',
    description = 'Benchmark face anonymization methods',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

parser.add_argument('original_dataset_dir', type=dir_path, help='the directory with original dataset')
parser.add_argument('anonymized_dataset_dir', type=dir_path, help='the directory with anonymized dataset')
parser.add_argument('--non_matching_pairs_filepath', type=file_exists, help='the file containing relative paths (relative to {original_dataset_dir}) of non-matching pairs, comma separated')
parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
parser.add_argument('--fp_rate', type=float, default=0.005, help='maximum false positivity rate for face reidentification, only used if non-matching pairs provided')
parser.add_argument('-o', '--output_dir', type=create_dir_if_non_existent, default='./benchmark_output', help='the folder in which the results will be stored')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print more information')
parser.add_argument('-g', '--use_gpu', action='store_true', default=False, help='use GPU instead of CPU')
parser.add_argument('-d', '--detector', choices=['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'], default='dlib', help='used underlying face detector')
parser.add_argument('-b', '--benchmarks', nargs='+',
    choices=POSSIBLE_BENCHMARKS,
    default=POSSIBLE_BENCHMARKS,
    help='specifies which benchmarks should run. If none specified, command runs the whole benchmark suite')

def compare_dataset_dirs(original_dir: str, anonymized_dir: str):
    """
    Compare dataset dirs. Check if:
    * all files are present in both folders
    * they're all images
    * the image sizes of each pair match
    """
    for entry in os.scandir(original_dir):
        if entry.is_dir():
            compare_dataset_dirs(entry.path, os.path.join(anonymized_dir, entry.name))
        elif entry.is_file():
            anonymized_file_path = os.path.join(anonymized_dir, entry.name)
            assert os.path.exists(anonymized_file_path) is True, f'Could not find a matching file for {entry.path}'
            assert filetype.is_image(entry.path) is True, f'{entry.path} is not an image'
            assert filetype.is_image(anonymized_file_path) is True, f'{anonymized_file_path} is not an image'
            img1 = Image.open(entry.path)
            img2 = Image.open(anonymized_file_path)
            assert img1.size == img2.size, f'{entry.path} and {anonymized_file_path} have different sizes'

if __name__ == '__main__':
    args = parser.parse_args()
    config = Configuration(
        original_dataset_dir=args.original_dataset_dir,
        anonymized_dataset_dir=args.anonymized_dataset_dir,
        non_matching_pairs_filepath=args.non_matching_pairs_filepath,
        use_gpu=args.use_gpu,
        batch_size=args.batch_size,
        fp_rate=args.fp_rate,
        output_dir=args.output_dir,
        detector=args.detector,
    )
    # Configure default log level based on verbose arg
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    LOG.info('Running AnonyBench')
    LOG.info(f'Using {"GPU" if config.use_gpu else "CPU"} if possible.')
    # Disable PIL logging to avoid log pollution
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # Check if dataset dirs have matching files
    # Possibly check if the images are same and if yes, print a warning
    LOG.debug('Comparing dataset directories')
    compare_dataset_dirs(config.original_dataset_dir, config.anonymized_dataset_dir)

    # Importing tf before torch helped with some GPU crashes...
    import tensorflow as tf
    if not config.use_gpu:
        tf.config.set_visible_devices([], 'GPU')
        tf.config.set_soft_device_placement(True)

    # Debug check for availability of GPUs on the system
    tf_gpu_avail = len(tf.config.list_physical_devices('GPU')) > 0
    import torch
    torch_gpu_avail = torch.cuda.is_available()
    LOG.debug(f'Is GPU available for TensorFlow? {tf_gpu_avail}')
    LOG.debug(f'Is GPU available for PyTorch? {torch_gpu_avail}')
    
    # Run benchmarks based on passed settings
    if 'face_detection' in args.benchmarks:
        LOG.info('Running face detection benchmark')
        run_face_detection_benchmark(config)

    if 'face_reidentification' in args.benchmarks:
        LOG.info('Running face reidentification benchmark')
        run_reidentify_faces_benchmark(config)

    if 'facial_attributes' in args.benchmarks:
        LOG.info('Running facial attributes benchmark')
        run_analyze_facial_attributes_benchmark(config)

    if 'gan_metrics' in args.benchmarks:
        LOG.info('Running GAN metrics benchmark')
        run_compute_gan_metrics_benchmark(config)

    if 'classifier' in args.benchmarks:
        LOG.info('Running classifier benchmark')
        run_classify_benchmark(config)
