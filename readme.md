##Problem understanding.

So in this problem we want to predict the sentiments of the movie review that the review is positive or not and we have provided with the labelled data.
so from above information we can conclude that this problem belongs to supervised classification category.the number of algorithms we can use for this in 
machine learning is random forest,SVM etc and in Deeplearning recurrent neural network with LSTM so here i am going to use LSTM algorithm.

##DATA COLLECTION
data is collected from kaggle.

##DATA PREPROCESSING

first we get the data and then preprocess it, by using stopwords library in nltk. stop words are those worrs which we filter out before feeding our data
into a model.And,the,you etc are consodered as stopwords. We also remove all apecial characters and numeric character because these have nothing to do with 
the sentimental analysis.then we join all of them again into words and make a string of words again.

##SPLITTING OF DATA SET

we split our data in train and validation part so that we can calculate take care of overfitting or underfitting by looking at the accuracies of train and
validation set.

##WORD Tokenisation.

we tokenize the words in numeric values and here we can choose how many words we want to tokenize, in my code i have choose 2000 so most frequenty comming 2000 words
will be tokenized.
**how tokenizer works**
The Tokenizer stores everything in the word_index during fit_on_texts. Then, when calling the texts_to_sequences method, only the top num_words which comes in
fits_on_text will be considered.
Then we pad the sequence, it is  used to ensure that all sequences in a list have the same length and change all of them to intizers. 
after this each review is an ordered array of integers, Make each review fixed size (say 400) in our case so shorter reviews get padded with 0’s in front and 
longer reviews get truncated to 200.

##MAKING MODEL

we import sequential from keras models. sequential model is a linear stack of layers.and we can create sequential model by passing a list to layer constructors in the layer
no we add embedding layer with 20000 words and funneling them into 128 hidden neurons.
then we added LSTM layer with 128 neurons and and dropout of 0.2 means 20% of neurons connection will de dropped out so that the neuron can't only relay on one feedback.
and recurrent dropout will drop the neurons between the networs it is also 0.2 in our case.

##model compilation
we compile our model with error function as "binary_crossentropy" and optimizer as adam optimizer and an accuracy matrix.
now our model is ready to be trained.

##model fitting

we fit out model with batch size of 32 and number of epoch is 5. after this we get out trained model. i have trained this for 3,5 and 8 epochs

we get best output at 5 epochs with a batch size of 32. the accuracy we got the F1 score here is 0.858169 and loss: 0.2469 - acc: 0.8999 - val_loss: 0.3791 - val_acc: 0.8594
and for 8 epochs with a batch size of 32. we got the F1 score here is 0.846318 and loss: 0.1955 - acc: 0.9228 - val_loss: 0.4001 - val_acc: 0.8482

##LINK OF KAGGLE
on kaggle i tried to explaind exlain every thing in detail sir chek that also.
i've uploaded this assignment on kaggle and the link of that is   https://www.kaggle.com/dhaka22/kernelffc6846ccb?scriptVersionId=6333399
i've done the same in jupyter notebook aslo code is provided in the file.





