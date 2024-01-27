<center><h1>Virtual Environment Installation</h1></center>

####  The third-party package required is specified in the requirements.txt file.

#### Below are some tips for installing several packages that may present challenges during setup.
#### Install any additional required Python packages as indicated by error messages.

### model weight

You can download the weight directory from https://drive.google.com/drive/folders/1xgoOgl7KIu_6fxy_vCYw7ZP11W75ZXiK.

### how to install Uni-Mol

1. You can use Conda as a Python environment manager to create a dedicated Python virtual environment for Uni-Mol.

```bash
conda create -n unimol python=3.10 pip
conda activate unimol
```

2. Uni-Mol relies on the high-performance distributed framework, Uni-Core, developed by DeepTech Science based on PyTorch. Therefore, you should first install Uni-Core. You can refer directly to the official code repository at [Uni-Core](https://github.com/dptech-corp/Uni-Core). Below is a possible configuration scheme that I used.

+ To install PyTorch, you can use the following command (my CUDA version is 11.3, please tailor this command to your own version)：

  ```bash
  pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
  ```

+ The Uni-Core repository provides several precompiled .whl files. For instance, if your CUDA version is 11.3, the torch version is 1.11.0, the Python version is 3.10, the operating system is Linux, and the hardware instruction set is x86_64, you can download the wheel file from [whl](https://github.com/dptech-corp/Uni-Core/releases/download/0.0.2/unicore-0.0.1+cu113torch1.11.0-cp310-cp310-linux_x86_64.whl)，For precompiled files for other environments, see the details at [releases](https://github.com/dptech-corp/Uni-Core/releases). 

  ```bash
  # Please tailor the command to your own version. 
  # download whl 文件：https://github.com/dptech-corp/Uni-Core/releases/download/0.0.2/unicore-0.0.1+cu113torch1.11.0-cp310-cp310-linux_x86_64.whl
  pip install unicore-0.0.1+cu113torch1.11.0-cp310-cp310-linux_x86_64.whl
  
  ```

3. install rdkit

Option 1: 

```bash
pip install rdkit-pypi
```

Option 2: 

```bash
conda install -y -c conda-forge rdkit==2021.09.5
```

4. Download the Uni-Mol code and proceed with the installation.

```bash
# first, download the source code [Uni-Mol](https://github.com/dptech-corp/Uni-Mol)
cd Uni-Mol/unimol
pip install .
```


### how to install torch_cluster, torch_scatter, torch_sparse, torch_spline_conv and torch_geometric
1. First, check the version of PyTorch.

```bash
python
import torch
print(torch.__version__)
print(torch.version.cuda)
```

2. Visit the official installation page for PyTorch Geometric at https://pytorch-geometric.com/whl/, and download the corresponding version of the wheels. (pay attention to the cuda version and torch version)

3. To install the PyTorch Geometric libraries such as torch-cluster, torch-geometric, torch-scatter, torch-sparse, and torch-spline-conv that match your torch and CUDA versions, you would use pip in the command line with the specific URLs you found for each library. Here's a general template for how you would do it, but you'll need to replace the <URL> placeholders with the actual URLs you found for the corresponding versions of the libraries:

```bash
pip install <torch_cluster_url>
pip install <torch_geometric_url>
pip install <torch_scatter_url>
pip install <torch_sparse_url>
pip install <torch_spline_conv_url>
```

Please ensure to replace each <URL> with the actual download link that matches your CUDA and torch versions from the PyTorch Geometric wheels page.

