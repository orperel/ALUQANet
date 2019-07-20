conda create -n aluqa python=3.6
source activate aluqa
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
pip install tb-nightly==1.15.0a20190616
pip install future
pip install allennlp
