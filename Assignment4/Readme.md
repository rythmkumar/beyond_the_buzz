Data is loaded using pandas <br>
Using nltk imported stopwords 
Then preprocessed data - 1. Removed all the non word characters , spaces and digits
Then tockenize the dataset for the words not in stopwords and took only the first 1000 most frequent words 
Now padding is done to make all the sentences of same length 
Now data is converted to tensor dataset and shuffled using dataloader
now lstm model is applied 
accuracy function is created . The predictions and label are vectors so they are squeezed to min dim.
Training function is created
Graph of Train loss vs validation loss is plotted 
Graph of Train accuracy vs validation accuracy is also plotted 

Finaly Sentiment analysis is done.
