import logging, os, shutil, tempfile, torch, pandas as pd
from tqdm import tqdm
from cleanfid import fid
from PIL import Image
from torchvision import transforms
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..cli import Configuration

LOG = logging.getLogger('GAN metrics')

def get_path_pairs(original_dataset_dir, anonymized_dataset_dir):
    """Get list of path pairs in the form of (source_path, anonymized_path)"""
    path_pairs = []
    for dirpath, _, fnames in os.walk(original_dataset_dir):
        for fname in fnames:
            source_file_path = os.path.join(dirpath, fname)
            rel_file_path = os.path.relpath(source_file_path, start=original_dataset_dir)
            path_pairs.append((source_file_path, os.path.join(anonymized_dataset_dir, rel_file_path)))
    return path_pairs

def compute_lpips(path_pairs, device, batch_size):
    """Compute the LPIPS metric based on pairs of images."""
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    lpips_vals = []
    for i in tqdm(range(0, len(path_pairs), batch_size), desc='LPIPS'):
        orig_images = torch.stack([transforms.ToTensor()(Image.open(ofp)) for ofp, _ in path_pairs[i:i + batch_size]]).to(device)
        anon_images = torch.stack([transforms.ToTensor()(Image.open(afp)) for _, afp in path_pairs[i:i + batch_size]]).to(device)
        with torch.no_grad():
            lpips_vals.append(float(lpips(orig_images, anon_images)))
    return sum(lpips_vals) / len(lpips_vals)

def compute_ssim(path_pairs, device, batch_size):
    """Compute the SSIM metric based on pairs of images."""
    ssim = StructuralSimilarityIndexMeasure().to(device)
    ssim_vals = []
    for i in tqdm(range(0, len(path_pairs), batch_size), desc='SSIM '):
        orig_images = torch.stack([transforms.ToTensor()(Image.open(ofp)) for ofp, _ in path_pairs[i:i + batch_size]]).to(device)
        anon_images = torch.stack([transforms.ToTensor()(Image.open(afp)) for _, afp in path_pairs[i:i + batch_size]]).to(device)
        with torch.no_grad():
            ssim_vals.append(float(ssim(orig_images, anon_images)))
    return sum(ssim_vals) / len(ssim_vals)


def run_compute_gan_metrics_benchmark(config: 'Configuration'):
    """Run GAN metrics benchmark."""
    device = torch.device('cuda') if config.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    path_pairs = get_path_pairs(config.original_dataset_dir, config.anonymized_dataset_dir)
    data = {
        'fid': [fid.compute_fid(config.original_dataset_dir, config.anonymized_dataset_dir, batch_size=config.batch_size, device=device)],
        # ! this needs 'pip install git+https://github.com/openai/CLIP.git'
        # 'clip-fid': [fid.compute_fid(original_dataset_dir, anonymized_dataset_dir, mode='clean', model_name='clip_vit_b_32', batch_size=batch_size, device=device)],
        'lpips': [compute_lpips(path_pairs, device, config.batch_size)],
        'ssim': [compute_ssim(path_pairs, device, config.batch_size)],
    }
    dir_basename = os.path.basename(os.path.normpath(config.anonymized_dataset_dir))
    output_path = os.path.join(config.output_dir, f'gan_metrics__{dir_basename}.csv')
    LOG.info(f'Saving results to {output_path}')
    pd.DataFrame.from_dict(data).to_csv(output_path)
