



Install instructions
----------------------------------
conda create -n eurosat2 python=3.7

conda activate eurosat2
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge opencv -y
conda install -c huggingface transformers -y
conda install -c conda-forge huggingface_hub -y
conda install -c huggingface -c conda-forge datasets -y
conda install -c intel scikit-learn -y
conda install -c anaconda seaborn -y

pip install psutil wget paramiko accelerate nvidia-ml-py3



