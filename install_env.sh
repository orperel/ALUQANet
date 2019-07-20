conda create -n aluqa python=3.6
source activate aluqa
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install tb-nightly
pip install future
pip install allennlp
