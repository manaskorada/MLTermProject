# 🏃 Running the LSTM
To run the LSTM, simply open the `LSTMwFeature.ipynb` file with jupiter notebook, and run the cells in order. There may be some dependencies issues, depending on what version of Numpy you are using. 
Gensim requires numpy 1.24.x, so it is recommended that you select that when installing the required libraries. THere should be information on how to do so 
in the notebook. To grab the datasets, go to the `IncreaseParameterScripts/` directory and download the `FINALtestUpdated.csv` and `FINALtrainUpdated.csv` files, and place them in your runtime. 

## 🔧 Architecture
The LSTM is implemented with Keras using 100 LSTM cells for the word vector embeddings, and 128 dense cells for the 21 other features. We then merge the two outputs and concatenate them into a final dense layer, then run that through a softmax to choose the class outcome.


|Layer (type)        | Output Shape      |    Param # | Connected to       |
|--------------------|-------------------|------------|--------------------|
| text_input  (InputLayer)    | (None, 80)        |          0 | -                 |
| embedding_5  (Embedding)       | (None, 80, 128)   |  5,011,840 | text_input[0][0]  |
| feature_input (InputLayer)      | (None, 21)        |          0 | -                 |
| lstm_5 (LSTM)       | (None, 100)       |     91,600 | embedding_5[0][0] |
| dense_11 (Dense)    | (None, 64)        |      1,408 | feature_input[0]… |
| concatenate_4 (Concatenate)      | (None, 164)       |          0 | lstm_5[0][0],     |
| dropout_2 (Dropout) | (None, 164)       |          0 | concatenate_4[0]… |
| dense_13 (Dense)    | (None, 4)         |        660 | dropout_2[0][0]   |

 Total params: 5,105,508 (19.48 MB)
 Trainable params: 5,105,508 (19.48 MB)
 Non-trainable params: 0 (0.00 B)
None

# 🏃 Running the MLP
To run the MLP, simply open the `FinalMLP.ipynb` file with jupiter notebook, and run the cells in order. We utilized Google Colab in this project.

## 🔧 Architecture
### Input Layer
Takes in a combined feature vector that merges:
TF-IDF features from the article's title and description (up to 1500 features).
21 engineered numerical features (e.g., sentiment scores, word counts, zero-shot scores, etc.).

### Hidden Layers
First Hidden Layer
Fully connected (Linear) layer with 512 neurons.
ReLU activation function.
Dropout (30%) to prevent overfitting.
Second Hidden Layer
Fully connected layer with 256 neurons (half of the previous).
ReLU activation again for non-linearity.

### Output Layer
Fully connected layer with 4 output units, one for each news class.
No activation applied here — handled internally by CrossEntropyLoss.

### Training Details
Loss Function: Cross Entropy
Optimizer: Adam
Learning Rate: 0.0005
Epochs: 20
Batch Size: 64

# 🏃 Running the CNN
To run the cnn, the train and test data must be already downloaded, and running the training cell allows you to upload them to the notebook. This approach was taken due to my colab notebook having pathing issues to my device, and I had to work around it.
I utilized the A100 GPU in colab for practical training times, as this process was often very lengthy across 20 Epochs, so it is important to set the logic to 'CPU' or the base available colab card if the user does NOT have colab pro. From there, run each cell in order and ensure runtime is complete before executing the next one.

### Input Layer
Takes in the features shaped by the number of numeric columns in the dataset.

### Blocks
Block 1, 2, and 3 have sizes of 32, 64, and 128 filters respectively, all with kernal size 3 and ReLU.

### Training details
Loss function: cross entropy
Optimizer: adam
Epochs: 20, with included early stopping
Batch size: 32
Validation split: 20%



