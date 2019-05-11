import csv
import re
import random
import numpy as np

from IPython import embed

#Separates a file with mixed positive and negative examples into two.
def separate_dataset(filename):
    good_out = open("good_"+filename,"w+");
    bad_out  = open("bad_"+filename,"w+");
    
    #+ This "seen" variable is used as a counter, which records the number of
    #+     available training data.
    seen = 1;
    with open(filename,'r') as f:
        
        line = f.readline()
        
        while line:
            seen +=1

            #+ Sentiment should be the first character, which is 0
            sentiment = line[0]
            
            #+ A sentence should be obtained like this
            sentence = line[2:]
            
            if (sentiment == "0"):
                bad_out.write(sentence)
            else:
                good_out.write(sentence)
            
            if (seen%10000==0):
                print(seen)
            
            #+ Read next sentence
            line = f.readline()

    good_out.close();
    bad_out.close();

def split(txt, seps):
    default_sep = seps[0]
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]

#Load Dataset
def get_dataset_str(goodfile,badfile,limit,randomize=True):
    good_x = split(goodfile, ("\n"))
    bad_x = split(badfile, ("\n"))
    
    
    if (randomize):
        random.shuffle(bad_x)
        random.shuffle(good_x)
    
    good_x = good_x[:limit]
    bad_x = bad_x[:limit]

    x = good_x + bad_x
    x = [clean_str(s) for s in x]
    
    
    positive_labels = [[1] for _ in good_x]
    negative_labels = [[0] for _ in bad_x]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x,y]




#Clean Dataset
def clean_str(string):
    
    
    #EMOJIS
    string = re.sub(r":\)","emojihappy1",string)
    string = re.sub(r":P","emojihappy2",string)
    string = re.sub(r":p","emojihappy3",string)
    string = re.sub(r":>","emojihappy4",string)
    string = re.sub(r":3","emojihappy5",string)
    string = re.sub(r":D","emojihappy6",string)
    string = re.sub(r" XD ","emojihappy7",string)
    string = re.sub(r" <3 ","emojihappy8",string)
    
    string = re.sub(r":\(","emojisad9",string)
    string = re.sub(r":<","emojisad10",string)
    string = re.sub(r":<","emojisad11",string)
    string = re.sub(r">:\(","emojisad12",string)
    
    #MENTIONS "(@)\w+"
    string = re.sub(r"(@)\w+","mentiontoken",string)
    
    #WEBSITES
    string = re.sub(r"http(s)*:(\S)*","linktoken",string)
    
    #STRANGE UNICODE \x...
    string = re.sub(r"\\x(\S)*","",string)
    
    #General Cleanup and Symbols
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()



#Generate random batches
#Source: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def gen_batch(data, batch_size, num_epochs, shuffle=True):
    """
        Generates a batch iterator for a dataset.
        """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



