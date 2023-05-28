# AnonyBench
AnonyBench is a suite of benchmarks for evaluation of face anonymization methods. The main goal is objective evaluation of face anonymization methods.

## What do you need
The only needed input from the user: 
- two folders, first with original dataset, second with anonymized dataset
- the folders have the same structure
- all the files are images
- there's one face in each of the images

We suggest using e.g. LFW or CelebA-HQ datasets.

## Installation

```bash
conda create --name anonybench python=3.8
conda activate anonybench
pip install -r benchmarks/requirements.txt
```

### GPU support
If you have CUDA installed and command `echo $LD_LIBRARY_PATH` gives you a path, you're most likely fine.  
If not, please set up CUDA using commands below:
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

How to test if you have correct GPU setup? Run:
```bash
python benchmarks/cli.py folder1 folder2 -v
```
You'll see two lines:  
`Is GPU available for TensorFlow? True/False` and `Is GPU available for PyTorch? True/False`.  
You want `True` for both of them.

## Using the CLI
If you want to run the full suite on our example, simply use:
```bash
python benchmarks/cli.py ./examples/lfw ./examples/lfw_deepprivacy2/ --non_matching_pairs_filepath ./examples/lfw_non_matching_pairs.txt
```
Add `-g` if you want to use a GPU.  
Add `-v` if you want to see output for debugging.

If you only want e.g. GAN metrics, you can run:
```bash
python benchmarks/cli.py ./examples/lfw ./examples/lfw_deepprivacy2/ -b gan_metrics
```

The simplest way to understand the CLI is to run the help command:
```bash
python benchmarks/cli.py -h
```

## Visualization
If you want to visualize your results, simply run:
```bash
python benchmarks/visualize.py
```
or run:
```bash
python benchmarks/visualize.py -f html
```
if you want an HTML file with plots included.  
To learn more about CLI options, use:
```bash
python benchmarks/visualize.py -h
```

## Citing
If you want to cite AnonyBench in your work, you can use:

```
@misc{anonybench_2023,
 author = {Moravčík, Jiří},
 title = {AnonyBench},
 year = {2023},
 howpublished = {\url{https://github.com/jirimoravcik/AnonyBench}}
}
```