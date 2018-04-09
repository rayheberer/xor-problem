import numpy as np
from sklearn.preprocessing import OneHotEncoder

def binary_equal_length(dataset_size=100000, sequence_length=50):
	data = np.random.binomial(1, 0.5, (dataset_size, sequence_length, 1))
	labels = np.reshape(np.sum(data, axis=1) % 2, (-1, 1))
	labels = OneHotEncoder().fit_transform(labels).toarray()

	lengths = np.ones((dataset_size, 1)) * sequence_length

	return data, labels, lengths

def binary_variable_length(dataset_size=100000, max_sequence_length=50):
	data = []
	labels = []

    #TODO: remove this loop
	for _ in range(dataset_size):
		n = np.random.randint(1, max_sequence_length+1)
		data_row = np.random.binomial(1, 0.5, (1, n))
		data.append(data_row)

		label_row = np.sum(data_row, axis=1) % 2
		labels.append(label_row)
	labels = np.array(labels)

	# TODO: buid lengths array

	return data, labels