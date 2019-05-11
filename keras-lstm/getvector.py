import clearndata
import numpy as np
from tensorflow.contrib import learn
import random
import csv
def readfile():
    pfile = "alexapos.csv"
    nfile = "alexaneg.csv"

    # Read pfile
    with open(pfile, 'r', encoding="unicode_escape") as content_file:
        content = content_file.read()
    p_data = content.splitlines()
    
    # Read nfile
    with open(nfile, 'r', encoding="unicode_escape") as content_file:
        content = content_file.read()

    n_data = content.splitlines()

    p_data2 = remove_label(p_data)
    n_data2 = remove_label(n_data)

    p_data3 = random_choice(p_data2,200)
    n_data3 = random_choice(n_data2,200)
    positive_sentences = return_string(p_data3)
    negative_sentences = return_string(n_data3)

    return positive_sentences, negative_sentences



            
    


def read_twitter():
    array = []
    with open("twetter1.txt", "r") as ins:
        for line in ins:
            array.append(line)


    pdata = []
    ndata = []
    for i in array:
        if "positive" in i:
            a = i.split("positive")
            if len(a) ==2:
                pdata += [a[1]]

        else:
            a = i.split("negative")
            if len(a) ==2:
                ndata += [a[1]]
    print(len(pdata))
    print(len(ndata))

    return pdata,ndata

def remove_label(mylist):
    new_list = []
    for d in mylist:
        new_list.append(d[2:])
    return new_list

def print_all(mylist):
    for d in mylist:
        print(d)
        print("")

def random_choice(mylist,num):
    new_list = []
    indices = random.sample(range(0, len(mylist)), num )
    for i in indices:
        new_list.append(mylist[i])
    return new_list

def return_string(mylist):
    my_string = ""
    for l in mylist:
        my_string += l + "\n"
    return my_string


def getvector():
    dev_size = 0.1
    
    ps2,ns = read_twitter()
    ps1 = random_choice(ps2,4500)
    ps = return_string(ps1)
    ns = return_string(ns)
    
    testx, testy = readfile()
    
    test, ytest = clearndata.get_dataset_str(ps,ns, 500000)
    x_text, y = clearndata.get_dataset_str(testx,testy, 500000)

    #x_text = x_text + test
    max_sentence_len =  max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_len)
    x = list(vocab_processor.fit_transform(x_text))
    x = np.array(x)
    print(len(x),123)
    #testx = np.array(x[-400:])
    #x = np.array(x[:-400])
    

    #randomly shuffle the vectors
    np.random.seed(42)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(dev_size * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return x_train,x_dev,y_train,y_dev,testx,testy

#getvector()

