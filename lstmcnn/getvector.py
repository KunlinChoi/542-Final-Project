import clearndata
import numpy as np
from tensorflow.contrib import learn
import random
import csv
def readfile(f1,f2,num1,num2):
    pfile = f1
    nfile = f2

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

    p_data3 = random_choice(p_data2,num1)
    n_data3 = random_choice(n_data2,num2)
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
        if len(d)> 15:
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

    testx, testy = readfile("moviepos.csv","movieneg.csv",10000,10000)
    ps,ns = readfile("alexapos.csv","alexaneg.csv",500,200)
    test, ytest = clearndata.get_dataset_str(ps,ns, 500000)
    x_text, y = clearndata.get_dataset_str(testx,testy, 500000)

    x1 = x_text+test
    #x_text = x_text + test
    max_sentence_len =  max([len(x.split(" ")) for x in x1])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_len)
    x = list(vocab_processor.fit_transform(x_text))
    x = np.array(x)
    testx = np.array(list(vocab_processor.fit_transform(test)))
    

    #randomly shuffle the vectors
    np.random.seed(42)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_sample_index = -1 * int(dev_size * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return x_train,x_dev,y_train,y_dev,testx,ytest,vocab_processor

#getvector()

