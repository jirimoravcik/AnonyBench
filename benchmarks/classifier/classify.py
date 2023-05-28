import logging, os, shutil, tempfile, torch, pandas as pd
import numpy as np
import torch.utils.data
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from typing import TYPE_CHECKING
from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
if TYPE_CHECKING:
    from ..cli import Configuration

SEED = 42
FOLDS = 5

LOG = logging.getLogger('Classifier')

class CustomDataSet(torch.utils.data.Dataset):
    """Used to fetch images."""
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = []
        for dirpath, _, fnames in os.walk(main_dir):
            for fname in fnames:
                source_file_path = os.path.join(dirpath, fname)
                rel_file_path = os.path.relpath(source_file_path, start=main_dir)
                self.all_imgs.append(rel_file_path)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.all_imgs[idx]
    
class SVMDataset(torch.utils.data.Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.vector_label_pairs = []
        for dirpath, _, fnames in os.walk(main_dir):
            for fname in fnames:
                if '.npy' not in fname:
                    continue
                source_file_path = os.path.join(dirpath, fname)
                rel_file_path = os.path.relpath(source_file_path, start=main_dir)
                self.vector_label_pairs.append((rel_file_path, 0 if 'orig' in rel_file_path else 1))

    def __len__(self):
        return len(self.vector_label_pairs)

    def __getitem__(self, idx):
        rel_path, label = self.vector_label_pairs[idx]
        vector_loc = os.path.join(self.main_dir, rel_path)
        vector = np.load(vector_loc)
        return vector, label

def run_classify_benchmark(config: 'Configuration'):
    device = torch.device('cuda') if config.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    resnet = InceptionResnetV1(pretrained='casia-webface', device=device).eval()

    # Create a temporary folder with the dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copytree(config.original_dataset_dir, os.path.join(tmpdir, 'orig/'), dirs_exist_ok=True)
        shutil.copytree(config.anonymized_dataset_dir, os.path.join(tmpdir, 'anon/'), dirs_exist_ok=True)
        dataset = CustomDataSet(tmpdir, transforms.Compose([
            transforms.ToTensor(),
        ]))
        
        for inputs, filenames in tqdm(torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, num_workers=4)):
            with torch.no_grad():
                features = resnet(inputs.to(device))
            for i in range(len(filenames)):
                feature, filename = features[i, :], filenames[i]
                feature_numpy = feature.cpu().numpy()
                path = os.path.join(tmpdir, filename.split('.')[0] + '.npy')
                np.save(path, feature_numpy / np.linalg.norm(feature_numpy))
        
        svm_dataset = SVMDataset(tmpdir)

        kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        kfold_acc = []
        for fold, (train_idx, test_idx) in enumerate(kfold.split(svm_dataset)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

            data_loaders = {
                'train': torch.utils.data.DataLoader(svm_dataset, batch_size=len(svm_dataset)*2, sampler=train_subsampler, num_workers=4),
                'test': torch.utils.data.DataLoader(svm_dataset, batch_size=len(svm_dataset)*2, sampler=test_subsampler, num_workers=4),
            }
            svm = LinearSVC()
            inputs, labels = next(data_loaders['train'].__iter__())
            X = inputs.numpy()
            y = labels.numpy()
            svm.fit(X, y)
            inputs, labels = next(data_loaders['test'].__iter__())
            X_test = inputs.numpy()
            y_test = labels.numpy()
            y_pred = svm.predict(X_test)
            kfold_acc.append((y_pred == y_test).sum() / y_test.shape[0])
        
        data = {
            'fold': list(range(FOLDS)),
            'accuracy': kfold_acc,
        }
        dir_basename = os.path.basename(os.path.normpath(config.anonymized_dataset_dir))
        output_path = os.path.join(config.output_dir, f'classifier__{dir_basename}.csv')
        LOG.info(f'Saving results to {output_path}')
        pd.DataFrame.from_dict(data).to_csv(output_path)
