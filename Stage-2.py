#!/usr/bin/env python
# coding: utf-8

# In[2]:


import zipfile
import os

# Specify the name of the uploaded zip file
uploaded_zip_file = 'deu-eng.zip'

# Specify the directory where you want to extract the contents
extracted_dir = 'C:\\Users\\SATHWIK'

# Create the target directory if it doesn't exist
if not os.path.exists(extracted_dir):
    os.makedirs(extracted_dir)

# Open the uploaded zip file
zip_file_path = os.path.join(os.getcwd(), uploaded_zip_file)
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the target directory
    zip_ref.extractall(extracted_dir)

print(f"Successfully extracted files to {extracted_dir}")


# In[3]:


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# In[4]:


# split a loaded document into pairs
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


# In[5]:


import re
import string
from unicodedata import normalize
from numpy import array

# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [re_punc.sub('', w) for w in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


# In[6]:


from pickle import dump  # Assuming you are using pickle for serialization

# ...

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into English-German pairs
pairs = to_pairs(doc)
# clean sentences
cleaned_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(cleaned_pairs, 'english-german.pkl')
# spot check
for i in range(100):
    print('[%s] => [%s]' % (cleaned_pairs[i, 0], cleaned_pairs[i, 1]))


# In[7]:


from pickle import load
from pickle import dump
from numpy.random import shuffle

# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('english-german.pkl')

# reduce dataset size
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]

# random shuffle
shuffle(dataset)

# split into train/test
train, test = dataset[:9000], dataset[9000:]

# save
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')


# In[8]:


# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')


# In[9]:


from tensorflow.keras.preprocessing.text import Tokenizer


# In[10]:


get_ipython().system('pip install --upgrade pip')




# In[11]:


get_ipython().system('pip install tensorflow')


# In[28]:


from tensorflow.keras.preprocessing.text import Tokenizer

# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
# Define and fit the tokenizer on your data
# Define and fit the tokenizer on your data


# In[29]:


# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


# In[30]:


# Load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# Prepare English tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])

# Check if the tokenizer is created successfully
if eng_tokenizer is not None and eng_tokenizer.word_index:
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % eng_length)

    # Prepare German tokenizer
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    
    # Check if the German tokenizer is created successfully
    if ger_tokenizer is not None and ger_tokenizer.word_index:
        ger_vocab_size = len(ger_tokenizer.word_index) + 1
        ger_length = max_length(dataset[:, 1])
        print('German Vocabulary Size: %d' % ger_vocab_size)
        print('German Max Length: %d' % ger_length)


# In[31]:


# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=10, padding='post')
    return X


# In[32]:


# one hot encode target sequence
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# In[ ]:





# In[33]:


from keras.preprocessing.sequence import pad_sequences


# In[34]:


from keras.utils import to_categorical


# In[35]:


# Prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)

# Prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)


# In[36]:


# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    
    return model


# In[37]:


# Import necessary libraries
from keras.models import Sequential
from keras.layers import Embedding, LSTM, RepeatVector, TimeDistributed, Dense
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

# Define the encode_sequences and encode_output functions
# ...

# Define the NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # summarize defined model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    
    return model

# Placeholder values, replace them with your actual values
src_vocab = 3552
tar_vocab = 2178
src_timesteps = 10
tar_timesteps = 10
n_units = 256

# Assuming you have previously defined and compiled the model using define_model function
model = define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units)

# Assuming you have your dataset defined, tokenizers created, and functions defined
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)

testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)
from keras.preprocessing.sequence import pad_sequences

# Assuming your sequences are stored in trainX and trainY
trainX_padded = pad_sequences(trainX, maxlen=10, padding='post')
trainY_padded = pad_sequences(trainY, maxlen=10, padding='post')


# Fit model
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)


# In[38]:


# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
# prepare german tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
# prepare data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])


# In[39]:


from keras.models import load_model

# Load the model
model = load_model('model.h5')


# In[40]:


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



# In[41]:


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[42]:


# generate target given source sequence
def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# In[50]:


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

# ...

# Evaluate the skill of the model
def evaluate_model(model, tokenizer, source, raw_dataset, max_length):
    actual, predicted = list(), list()

    # Handle non-numeric elements in the sequences
    replacement_value = 0
    numeric_sequences = [
        [int(str(num)) if str(num).isdigit() else replacement_value for num in seq] for seq in source
    ]

    # Pad all sequences
    padded_sequences = pad_sequences(numeric_sequences, maxlen=10, padding='post')

    for i, source_seq in enumerate(padded_sequences):
        # translate encoded source text
        source_seq = source_seq.reshape((1, source_seq.shape[0]))
        translation = predict_sequence(model, tokenizer, source_seq)
        raw_target, raw_src, additional_info = raw_dataset[i]
        if i < 10:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())

    # flatten the lists of lists into a single list of strings
    actual_flat = [word for sublist in actual for word in sublist]
    predicted_flat = [word for sublist in predicted for word in sublist]

    # calculate BLEU score with smoothing
    smoothing_function = SmoothingFunction().method1
    bleu_1 = sentence_bleu([actual_flat], predicted_flat, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_function)
    bleu_2 = sentence_bleu([actual_flat], predicted_flat, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
    bleu_3 = sentence_bleu([actual_flat], predicted_flat, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smoothing_function)
    bleu_4 = sentence_bleu([actual_flat], predicted_flat, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)

    # print BLEU scores
    print('BLEU-1: %f' % bleu_1)
    print('BLEU-2: %f' % bleu_2)
    print('BLEU-3: %f' % bleu_3)
    print('BLEU-4: %f' % bleu_4)

# Assuming you have defined max_length somewhere before calling this function

# Call the evaluate_model function with the model, tokenizer, sources, raw_dataset, and max_length
evaluate_model(model, eng_tokenizer, trainX, raw_dataset, max_length)


# In[ ]:


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])

if eng_tokenizer is not None and eng_tokenizer.word_index:
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])

    # prepare german tokenizer
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = max_length(dataset[:, 1])

    # prepare data
    trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
    testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])

    # load model
    model = load_model('model.h5')

    # test on some training sequences
    print('train')
    evaluate_model(model, eng_tokenizer, trainX, train, raw_dataset)

    # test on some test sequences
    print('test')
    evaluate_model(model, eng_tokenizer, testX, test, raw_dataset)
else:
    print("Error: The English tokenizer was not created successfully.")


# In[ ]:




