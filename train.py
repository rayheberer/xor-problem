import sys
import model, data

random_seed = 41
dataset_size = 100000
max_sequence_length = 50
num_classes = 2

def main(lstm_units):
    data1, labels1, lengths1 = data.binary_equal_length(dataset_size, max_sequence_length)
    LSTM = model.LSTM(lstm_units)
    model.train_model(LSTM, data1, labels1, lengths1)
    return LSTM

if __name__ == "__main__":
    
    lstm_units = int(sys.argv[1])

    trained_model = main(lstm_units)