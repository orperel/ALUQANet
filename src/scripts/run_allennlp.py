from allennlp.commands import main  # pylint: disable=wrong-import-position

command = 'train'
include_package = 'bert-drop'
experiment = 'experiments/single_epoch_nabert.json'
output_path = 'results/nabert_single_epoch'
is_force_override_existing_results = True

force = '--force' if is_force_override_existing_results else ''
main()
# main(prog=f'allennlp {command} {experiment} -s {output_path} --include-package {include_package} {force}')