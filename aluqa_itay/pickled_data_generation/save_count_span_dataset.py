import pickle
from allennlp.common import Params
from allennlp.data import DatasetReader
from aluqa_itay.data_processing_generated import BertDropTokenizer, BertDropTokenIndexer, BertDropReader


def create_nabert_drop_reader(data_split='dev'):
    config = Params.from_file("span_data_generation.json")

    if data_split == 'dev':
        # data_path = "/specific/netapp5_2/gamir/advml19/perel/Drop/data/drop_dataset/drop_dataset_dev.json"
        data_path = "/home/itaysofer/Desktop/ALUQANet/data/generated_dataset/generated_dev.json"
    elif data_split == 'train':
        # data_path = "/specific/netapp5_2/gamir/advml19/perel/Drop/data/drop_dataset/drop_dataset_train.json"
        data_path = "/home/itaysofer/Desktop/ALUQANet/data/generated_dataset/generated_train.json"
    else:
        raise ValueError(f'Unsupported datasplit {data_split}, please use "dev" or "train"')

    config_params = config['dataset_reader']

    data_reader = DatasetReader.from_params(config_params)
    instances = data_reader.read(data_path)
    return instances

instances = create_nabert_drop_reader('train')
pickle.dump(instances, open("./pickled_data/generated_dataset_train.pickle", "wb+"))

instances = create_nabert_drop_reader('dev')
pickle.dump(instances, open("./pickled_data/generated_dataset_dev.pickle", "wb+"))
