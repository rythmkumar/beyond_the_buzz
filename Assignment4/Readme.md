Data is loaded using pandas <br>
Using nltk imported stopwords <br>
Then preprocessed data - 1. Removed all the non word characters , spaces and digits <br>
Then tockenize the dataset for the words not in stopwords and took only the first 1000 most frequent words <br>
Now padding is done to make all the sentences of same length <br>
Now data is converted to tensor dataset and shuffled using dataloader <br>
now lstm model is applied <br>
accuracy function is created . The predictions and label are vectors so they are squeezed to min dim. <br>
Training function is created<br>
Graph of Train loss vs validation loss is plotted <br>
Graph of Train accuracy vs validation accuracy is also plotted <br>
<br>
Finaly Sentiment analysis is done.<br>
