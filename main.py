from Seq2Seq import Seq2Seq,data_generator
import numpy as np

train_houses = [2,3,4,5,6]
test_house = 1
#appliance = 'washer_dryer' 
appliance = 'refrigerator' 
windows_length = 600
epochs = 15

#Type = 'CNN-RNN'
#Type = 'ConvLSTM'
#Type = 'Dense'
#Type = 'CNN-2d'   # 适用 mode：'od' 
#Type = 'CNN-1d-2'
Type = 'CNN(Chaoyun)'

seq2seq = Seq2Seq('o')
seq2seq.get_houses_data(train_houses,test_house,
                        appliance,windows_length,2000,200)
seq2seq.build_network(Type)
#seq2seq.load_model_and_history(Type)

seq2seq.train(epochs=epochs)
seq2seq.save_model_and_history()
seq2seq.plot_training_history()
seq2seq._demo_show()

