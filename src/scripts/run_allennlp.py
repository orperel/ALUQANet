from allennlp.commands import main  # pylint: disable=wrong-import-position

command = 'train'
experiment = 'experiments/naqanet.json'
output_path = 'results'
is_force_override_existing_results = True

force = '--force' if is_force_override_existing_results else ''
main(prog=f'allennlp {command} {experiment} -s {output_path} --include-package custom_models {force}')