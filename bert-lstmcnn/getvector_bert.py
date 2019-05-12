import clearndata
import numpy as np
from tensorflow.contrib import learn
from bert_serving.client import BertClient
import random
"""
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
                pdata += [a[1].strip("\t")]

        else:
            a = i.split("negative")
            if len(a) ==2:
                ndata += [a[1].strip("\t")]
    print(pdata[2])

    return pdata[:300],ndata[:300]
"""

def readfile():
    pfile = "moviepos.csv"
    nfile = "movieneg.csv"

    # Read pfile
    with open(pfile, 'r', encoding="unicode_escape") as content_file:
        content = content_file.read()
    p_data = content.splitlines()
    #print(p_data)
    # Read nfile
    with open(nfile, 'r', encoding="unicode_escape") as content_file:
        content = content_file.read()

    n_data = content.splitlines()

    p_data2 = remove_label(p_data)
    n_data2 = remove_label(n_data)
    print(p_data2)
    p_data3 = random_choice(p_data2)
    n_data3 = random_choice(n_data2)

    print("hello")
    return p_data3, n_data3

def remove_label(mylist):
    new_list = []
    for d in mylist:
        new_list.append(d[2:])
    return new_list

def print_all(mylist):
    for d in mylist:
        print(d)
        print("")

def random_choice(mylist):
    NUM_CHOICES = 10000
    new_list = []
    indices = random.sample(range(0, len(mylist)), NUM_CHOICES )
    for i in indices:
        new_list.append(mylist[i])
    return new_list

def return_string(mylist):
    my_string = ""
    for l in mylist:
        my_string += l + "\n"
    return my_string




def getvector_bert():
    p_text,n_text = readfile()
    #ptwitter,ntwitter = read_twitter()
    
    yp = [[1,0]]*len(p_text)
    yn = [[0,1]]*len(n_text)
    
    x_text = np.array(p_text + n_text)
    y = yp + yn
    y = np.array(y)


    #randomly shuffle the vectors
    state = np.random.get_state()
    np.random.shuffle(x_text)
    np.random.set_state(state)
    np.random.shuffle(y)

    print(type(x_text))

    x_text= list(x_text)


    xt = x_text

    #x1 = x_text + xt
    #print(len(x1))
    x1 = xt
    bc = BertClient(ip='localhost')
    
    x = bc.encode(x1)
    print(len(x),1)
    print(len(x[0]),2)
    print(len(x[0][0]),3)

    fw = open("vector.txt", 'w')

    for i in x:
        temp = list(i)
        temp2 = str(temp)
        fw.write(temp2)
        fw.write("\n")

    fy = open("label.txt", 'w')

    for i in y:
        temp = list(i)
        temp2 = str(temp)
        fy.write(temp2)
        fy.write("\n")
    
    x_train = x[:18000]
    y_train = y[:18000]
    x_dev = x[18000:]
    y_dev = y[18000:]

    #x_tp_dev = x[10000:10300]
    #x_tn_dev = x[10300:]

    #print(x_train,"a",y_train,"b",x_dev,"c",y_dev,"d", x_tp_dev,"e", x_tn_dev,file = open("log.txt","a"))
    return x_train,x_dev,y_train,y_dev#,x_tp_dev,x_tn_dev

getvector_bert()
