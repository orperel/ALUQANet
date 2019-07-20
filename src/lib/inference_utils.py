from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset import Batch
from allennlp.common.util import prepare_environment
from allennlp.models.archival import load_archive
from allennlp.nn.util import move_to_device


def load_model(model_path, weights_path):
    archive = load_archive(archive_file=model_path, cuda_device=0, weights_file=weights_path)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()
    return model, config


def create_drop_reader(config, data_split='dev', lazy=True):
    if data_split == 'dev':
        config_params = config['validation_dataset_reader']
        data_path = config['validation_data_path']
    elif data_split == 'train':
        config_params = config['dataset_reader']
        data_path = config['train_data_path']
    else:
        raise ValueError(f'Unsupported datasplit {datasplit}, please use "dev" or "train"')

    data_reader = DatasetReader.from_params(config_params)
    data_reader.lazy = lazy
    instances = data_reader.read(data_path)
    return instances


def data_instance_to_model_input(instance, model):
    dataset = Batch([instance])
    dataset.index_instances(model.vocab)
    cuda_device = model._get_prediction_device()
    model_input = move_to_device(dataset.as_tensor_dict(), cuda_device=cuda_device)
    return model_input
