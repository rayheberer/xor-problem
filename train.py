import sys
import model, data

random_seed = 41
dataset_size = 100000
max_sequence_length = 50
num_classes = 2

def main(lstm_units, epochs):
    print('Generating data...')
    data1, labels1, lengths1 = data.binary_equal_length(dataset_size, max_sequence_length)
    
    print('Building model...')
    LSTM = model.LSTM(lstm_units)
    print('{} unit LSTM built'.format(lstm_units))

    print('Training...')
    model.train_model(LSTM, data1, labels1, lengths1, epochs=epochs)
    return LSTM

if __name__ == "__main__":
    
    lstm_units = int(sys.argv[1]) if sys.argv[1] else 50

    epochs = int(sys.argv[2]) if sys.argv[2] else 1

    trained_model = main(lstm_units, epochs)