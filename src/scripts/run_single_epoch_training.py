from allennlp.commands.train import train_model_from_file


# See: https://github.com/allenai/allennlp/blob/master/allennlp/commands/train.py#L102
train_model_from_file(parameter_filename='experiments/single_epoch_naqanet.json',
                      serialization_dir='results_single',
                      force=True)

