# Neural-NILM
An Non-Intrusive Load Monitoring (NILM) method (also called Non-Intrusive Load Disaggregation) based on Neural Network. A sequence-to-sequence model and a sequence-to-point model are proposed. The project is based on Keras and the tensorflow version is on the way.

## Files
```
|-- Seq2Seq.py            // define a Seq2Seq Class
|-- Seq2point.py          // define a Seq2point Class
|-- model.py              // define some neural networks
|-- lib.py                // Some necessary helper functions
|-- main.py               // main function
```
## Neural network framework
Keras with tensorflow as backend

## Data set 数据集
REDD： http://redd.csail.mit.edu/    
由于训练用到的数据约为2G，故未上传。需要请前往上面链接自行下载。   
Because the data used in the training is about 2 G, it is not uploaded. Please go to the above link and download it yourself.

## Usage 使用
从主函数中调用各个文件并读取训练数据，即可开始训练。训练之后会自动将验证集上表现最好的模型及权值文件保存下来，训练曲线等数据保存为CSV文件。   

Call each file from the main function and read the training data to start the training. After training, the best performance model and weight files on the validation dataset will be saved automatically, and the training history and other data will be saved as CSV files.

