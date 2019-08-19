import os
import pickle

with open(os.path.join("./pickled_data/drop_dataset_train_1.pickle"), 'rb') as dataset_file1:
    instances1 = pickle.load(dataset_file1)

with open(os.path.join("./pickled_data/drop_dataset_train.pickle"), 'rb') as dataset_file:
    instances = pickle.load(dataset_file)

output_instances = instances1 + instances
pickle.dump(output_instances, open("./pickled_data/drop_dataset_train_joined.pickle", "wb+"))
