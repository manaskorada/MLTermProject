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

