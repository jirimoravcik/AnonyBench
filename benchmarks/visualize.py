import argparse, logging, os, pandas as pd, seaborn as sns, torch, io, base64, time
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from tabulate import tabulate
from sklearn.metrics import confusion_matrix

LOG = logging.getLogger('Visualization')

def create_dir_if_non_existent(string):
    """Check if dir exists. If not, attempt to create it."""
    if os.path.isdir(string):
        return string
    elif not os.path.exists(string):
        os.makedirs(string)
        return string

def get_face_detection_table(file_paths):
    datasets = ['anonymized', 'boxed', 'blurred', 'pixelized', 'blurred-full']
    headers = ['Method', 'Anonymized(↑)', 'Boxed(↑)', 'Blurred(↑)', 'Pixelized(↑)', 'Full blur(↑)']
    table = []
    for file_path in file_paths:
        anonymized_dir_name = os.path.basename(file_path).replace('.csv', '').split('__')[1]
        df = pd.read_csv(file_path)
        df = df[df['original'] == 'FOUND'] # Only evaluate images with faces detected on the original
        row = [anonymized_dir_name]
        for dataset in datasets:
            found = df.value_counts(dataset).get('FOUND', 0)
            not_found = df.value_counts(dataset).get('NOT_FOUND', 0)
            perc = round((found / (found + not_found)) * 100, 2)
            row.append(f'{perc:.2f}%')
        table.append(row)
    return table, headers

def get_face_reidentification_table(file_paths):
    headers = ['Method', 'Re-identified faces(↓)']
    table = []
    for file_path in file_paths:
        anonymized_dir_name = os.path.basename(file_path).replace('.csv', '').split('__')[1]
        df = pd.read_csv(file_path)
        perc = round(((df[df['verified'] == True].shape[0]) / df.shape[0]) * 100, 2)
        row = [anonymized_dir_name, f'{perc:.2f}%']
        table.append(row)
    return table, headers

def get_facial_attributes_table(file_paths):
    headers = ['Method', 'Mean absolute age difference(↓)', 'Gender preservation(↑)', 'Race preservation(↑)']
    table = []
    for file_path in file_paths:
        anonymized_dir_name = os.path.basename(file_path).replace('.csv', '').split('__')[1]
        df = pd.read_csv(file_path)
        genders = sorted(set([*(df['orig_dominant_gender'].unique()), *(df['anon_dominant_gender'].unique())]))
        races = sorted(set([*(df['orig_dominant_race'].unique()), *(df['anon_dominant_race'].unique())]))
        row = [anonymized_dir_name]
        df['age_diff'] = abs(df['orig_age'] - df['anon_age'])
        age_diff_mean, age_diff_std = df['age_diff'].mean(), df['age_diff'].std()
        row.append(f'{age_diff_mean:.2f} ± {age_diff_std:.2f}')
        cm = confusion_matrix(df['orig_dominant_gender'], df['anon_dominant_gender'])
        unchanged_gender_vals = []
        for i in range(len(genders)):
            unchanged_gender_vals.append(cm[i, i] / cm[i, :].sum())
        unchanged_gender = sum(unchanged_gender_vals) / len(unchanged_gender_vals)
        row.append(f'{(unchanged_gender*100):.2f}%')
        cm = confusion_matrix(df['orig_dominant_race'], df['anon_dominant_race'])
        unchanged_race_vals = []
        for i in range(len(races)):
            unchanged_race_vals.append(cm[i, i] / cm[i, :].sum())
        unchanged_race = sum(unchanged_race_vals) / len(unchanged_race_vals)
        row.append(f'{(unchanged_race*100):.2f}%')
        table.append(row)
    return table, headers

def get_race_confusion_matrices(file_paths):
    matrices = []
    for file_path in file_paths:
        anonymized_dir_name = os.path.basename(file_path).replace('.csv', '').split('__')[1]
        df = pd.read_csv(file_path)
        races = sorted(set([*(df['orig_dominant_race'].unique()), *(df['anon_dominant_race'].unique())]))
        idx = 0
        race_to_idx = {}
        for race in races:
            race_to_idx[race] = idx
            idx += 1
        confmat = ConfusionMatrix(task="multiclass", num_classes=len(races))
        confmat_rel = ConfusionMatrix(task="multiclass", num_classes=len(races), normalize='true')
        cm = confmat(
            torch.Tensor(list(map(race_to_idx.get, df['anon_dominant_race']))),
            torch.Tensor(list(map(race_to_idx.get, df['orig_dominant_race']))),
        )
        cm_rel = confmat_rel(
            torch.Tensor(list(map(race_to_idx.get, df['anon_dominant_race']))),
            torch.Tensor(list(map(race_to_idx.get, df['orig_dominant_race']))),
        )
        
        fig, axes = plt.subplots(ncols=2, figsize=(17, 6))
        sns.heatmap(cm, xticklabels=races, yticklabels=races, 
                    annot=True, fmt='g', ax=axes[0])
        sns.heatmap(cm_rel, xticklabels=races, yticklabels=races, 
                    annot=True, fmt='.2%', ax=axes[1])
        for ax in axes:
            ax.set_xlabel('Anonymized race')
            ax.set_xticks([i - 0.5 for i in range(1, len(races) + 1)], races, rotation=45, ha='right')
            ax.set_ylabel('Original race')
        axes[0].set_title(f'Race conf. matrix for {anonymized_dir_name}')
        axes[1].set_title(f'Relative race conf. matrix for {anonymized_dir_name}')
        img = io.BytesIO()
        fig.savefig(img, format='png',
                    bbox_inches='tight')
        img.seek(0)
        plt.close()
        matrices.append(base64.b64encode(img.getvalue()))
    return matrices

def get_gender_confusion_matrices(file_paths):
    matrices = []
    for file_path in file_paths:
        anonymized_dir_name = os.path.basename(file_path).replace('.csv', '').split('__')[1]
        df = pd.read_csv(file_path)
        genders = sorted(set([*(df['orig_dominant_gender'].unique()), *(df['anon_dominant_gender'].unique())]))
        idx = 0
        gender_to_idx = {}
        for race in genders:
            gender_to_idx[race] = idx
            idx += 1
        confmat = ConfusionMatrix(task="multiclass", num_classes=len(genders))
        confmat_rel = ConfusionMatrix(task="multiclass", num_classes=len(genders), normalize='true')
        cm = confmat(
            torch.Tensor(list(map(gender_to_idx.get, df['anon_dominant_gender']))),
            torch.Tensor(list(map(gender_to_idx.get, df['orig_dominant_gender']))),
        )
        cm_rel = confmat_rel(
            torch.Tensor(list(map(gender_to_idx.get, df['anon_dominant_gender']))),
            torch.Tensor(list(map(gender_to_idx.get, df['orig_dominant_gender']))),
        )
        fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
        sns.heatmap(cm, xticklabels=genders, yticklabels=genders, 
                    annot=True, fmt='g', ax=axes[0])
        sns.heatmap(cm_rel, xticklabels=genders, yticklabels=genders, 
                    annot=True, fmt='.2%', ax=axes[1])
        for ax in axes:
            ax.set_xlabel('Anonymized gender')
            ax.set_ylabel('Original gender')
        axes[0].set_title(f'Gender conf. matrix for {anonymized_dir_name}')
        axes[1].set_title(f'Relative gender conf. matrix for {anonymized_dir_name}')
        img = io.BytesIO()
        fig.savefig(img, format='png',
                    bbox_inches='tight')
        img.seek(0)
        plt.close()
        matrices.append(base64.b64encode(img.getvalue()))
    return matrices

def get_age_diff_histograms(file_paths):
    hists = []
    for file_path in file_paths:
        anonymized_dir_name = os.path.basename(file_path).replace('.csv', '').split('__')[1]
        df = pd.read_csv(file_path)
        fig, ax = plt.subplots()
        age_diff = df['orig_age'] - df['anon_age']
        bins = age_diff.max() - age_diff.min()
        age_diff.hist(ax=ax, bins=bins, align='left')
        ax.set_xlabel('Age difference')
        ax.set_ylabel('# of occurrences')
        ax.set_title(f'Age difference (original minus anonymized) for {anonymized_dir_name}')
        img = io.BytesIO()
        fig.savefig(img, format='png',
                    bbox_inches='tight')
        img.seek(0)
        hists.append(base64.b64encode(img.getvalue()))
    return hists

def get_gan_metrics_table(file_paths):
    metrics = ['fid', 'lpips', 'ssim']
    headers = ['Method', 'FID(↓)', 'LPIPS(↓)', 'SSIM(↑)']
    table = []
    for file_path in file_paths:
        anonymized_dir_name = os.path.basename(file_path).replace('.csv', '').split('__')[1]
        df = pd.read_csv(file_path)
        row = [anonymized_dir_name]
        for metric in metrics:
            row.append(f'{df[metric][0]:.4f}')
        table.append(row)
    return table, headers

def get_classifier_table(file_paths):
    headers = ['Method', 'Accuracy(↓)']
    table = []
    for file_path in file_paths:
        anonymized_dir_name = os.path.basename(file_path).replace('.csv', '').split('__')[1]
        df = pd.read_csv(file_path)
        row = [anonymized_dir_name]
        row.append(f'{df["accuracy"].mean():.4f}')
        table.append(row)
    return table, headers

def get_html_representation(html_tables, base64_plots):
    face_detection = '' if html_tables.get('face_detection') is None else f"""
        <h2>Face detection</h2>
        <p>We report percentages of detected faces. The baseline dataset are all the images from the original dataset in which we were able to detect the face. We use the face detector from Dlib.</p>
        {html_tables['face_detection']}
    """
    face_reidentification = '' if html_tables.get('face_reidentification') is None else f"""
        <h2>Face re-identification</h2>
        <p>We report percentages of re-identified faces based of false positive rate if non-matching pairs are provided, otherwise, we use a reasonable default threshold. We use ArcFace.</p>
        {html_tables['face_reidentification']}
    """
    conf_mat_race_imgs = []
    for conf_mat in base64_plots.get('conf_mat_race', []):
        conf_mat_race_imgs.append('<img src="data:image/png;base64, {}">'.format(conf_mat.decode('utf-8')))
    conf_mat_gender_imgs = []
    for conf_mat in base64_plots.get('conf_mat_gender', []):
        conf_mat_gender_imgs.append('<img src="data:image/png;base64, {}">'.format(conf_mat.decode('utf-8')))
    age_diff_hists = []
    for hist in base64_plots.get('age_diff_hist', []):
        age_diff_hists.append('<img src="data:image/png;base64, {}">'.format(hist.decode('utf-8')))
    facial_attributes = '' if html_tables.get('facial_attributes') is None else f"""
        <h2>Facial attributes</h2>
        {html_tables['facial_attributes']}
        <div style="text-align:center;">
        {'<hr>'.join(conf_mat_race_imgs)}
        <hr>
        {'<hr>'.join(conf_mat_gender_imgs)}
        <hr>
        {'<hr>'.join(age_diff_hists)}
        <hr>
        </div>
    """
    gan_metrics = '' if html_tables.get('gan_metrics') is None else f"""
        <h2>GAN metrics</h2>
        {html_tables['gan_metrics']}
    """
    classifier = '' if html_tables.get('classifier') is None else f"""
        <h2>Classifier</h2>
        {html_tables['classifier']}
    """
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AnonyBench results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet" crossorigin="anonymous">
  </head>
  <body style="padding: 16px;">
    <h1>AnonyBench results</h1>
    <p>Generated on: {time.strftime("%Y-%m-%d %H:%M")}</p>
    <hr>
    {face_detection}
    {face_reidentification}
    {facial_attributes}
    {gan_metrics}
    {classifier}
  </body>
</html>
    """.replace('<table>', '<table class="table table-bordered">')

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(
    prog = 'AnonyBench visualization',
    description = 'Visualize results from AnonyBench',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-s', '--source_dir', type=dir_path, default='./benchmark_output', help='the folder in which the results are stored')
parser.add_argument('-f', '--format', choices=['txt', 'html', 'latex'], default='txt', help='the format of the output file')
parser.add_argument('-d', '--dir', type=create_dir_if_non_existent, default='./visualization_output', help='the name of the output folder')
parser.add_argument('-o', '--output', default='benchmark_visualization', help='the name of the output file without extension')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print more information')
parser.add_argument('-n', '--name_contains', default=None, help='optional string that should be contained in file names to visualize')

if __name__ == '__main__':
    args = parser.parse_args()
    # Initialize logging
    logging.basicConfig()
    # Configure default log level based on verbose arg
    LOG.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    file_paths = {
        'face_detection': [],
        'face_reidentification': [],
        'facial_attributes': [],
        'gan_metrics': [],
        'classifier': [],
    }
    for file in os.scandir(args.source_dir):
        if file.is_file():
            if args.name_contains is not None and args.name_contains not in file.name:
                continue
            for key in file_paths.keys():
                if file.name.startswith(key):
                    file_paths[key].append(file.path)
    for key in file_paths:
        file_paths[key].sort()
    output_file_name = f'{args.output}.{args.format}'
    txt_representation = 'AnonyBench results\n------------------\n'
    latex_representation = 'AnonyBench results'
    html_tables = {}
    base64_plots = {}
    output_path = os.path.join(args.dir, output_file_name)
    table_format = 'fancy_grid'
    with open(output_path, 'w') as f:
        if len(file_paths['face_detection']) > 0:
            table, headers = get_face_detection_table(file_paths['face_detection'])
            txt_representation += f"""Percentages of detected faces\n{tabulate(table, headers, tablefmt='fancy_grid')}\n"""
            latex_representation += f"""Percentages of detected faces\n{tabulate(table, headers, tablefmt='latex')}\n"""
            html_tables['face_detection'] = tabulate(table, headers, tablefmt='unsafehtml')

        if len(file_paths['face_reidentification']) > 0:
            table, headers = get_face_reidentification_table(file_paths['face_reidentification'])
            txt_representation += f"""Percentages of re-identified faces\n{tabulate(table, headers, tablefmt='fancy_grid')}\n"""
            latex_representation += f"""Percentages of re-identified faces\n{tabulate(table, headers, tablefmt='latex')}\n"""
            html_tables['face_reidentification'] = tabulate(table, headers, tablefmt='unsafehtml')

        if len(file_paths['facial_attributes']) > 0:
            table, headers = get_facial_attributes_table(file_paths['facial_attributes'])
            txt_representation += f"""Facial attributes\n{tabulate(table, headers, tablefmt='fancy_grid')}\n"""
            latex_representation += f"""Facial attributes\n{tabulate(table, headers, tablefmt='latex')}\n"""
            html_tables['facial_attributes'] = tabulate(table, headers, tablefmt='unsafehtml')
            base64_plots['conf_mat_race'] = get_race_confusion_matrices(file_paths['facial_attributes'])
            base64_plots['conf_mat_gender'] = get_gender_confusion_matrices(file_paths['facial_attributes'])
            base64_plots['age_diff_hist'] = get_age_diff_histograms(file_paths['facial_attributes'])

        if len(file_paths['gan_metrics']) > 0:
            table, headers = get_gan_metrics_table(file_paths['gan_metrics'])
            txt_representation += f"""GAN metrics\n{tabulate(table, headers, tablefmt='fancy_grid')}\n"""
            latex_representation += f"""GAN metrics\n{tabulate(table, headers, tablefmt='latex')}\n"""
            html_tables['gan_metrics'] = tabulate(table, headers, tablefmt='unsafehtml')

        if len(file_paths['classifier']) > 0:
            table, headers = get_classifier_table(file_paths['classifier'])
            txt_representation += f"""Classifier\n{tabulate(table, headers, tablefmt='fancy_grid')}\n"""
            latex_representation += f"""Classifier\n{tabulate(table, headers, tablefmt='latex')}\n"""
            html_tables['classifier'] = tabulate(table, headers, tablefmt='unsafehtml')

        if args.format == 'txt':
            f.write(txt_representation)
        elif args.format == 'html':
            f.write(get_html_representation(html_tables, base64_plots))
        elif args.format == 'latex':
            f.write(latex_representation)

    LOG.info(f'Saving results to {output_path}')

    print(txt_representation)
