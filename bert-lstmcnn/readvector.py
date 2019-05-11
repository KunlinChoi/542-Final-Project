import numpy as np
def readvec():
    vec = []

    f = open('movalesmall.vec')
    for line in f:
        a = line.split()
        temp = []
        for i in a:
            a = float(i)
            temp += [a]
        vec += [temp]
    f.close()

    ret = []
    for i in vec:
        new = []
        for j in range(256):
            temp = []
            start = j*768
            end = j*768 + 768
            while start < end:
                temp += [float(i[start])]
                start += 1
            new += [temp]
        ret += [new]
    
    #print(ret[0][0][1])
    print(len(ret[0]))
    print(len(ret))
    return ret
#readvec()

def shuffle():
    data = readvec()
    trainneg = data[:1000]
    trainpos = data[1000:2000]
    trainy = [[0,1]] * len(trainneg) + [[1,0]] *len(trainpos)
    trainy = np.array(trainy)
    trainx = trainneg +trainpos
    trainx = np.array(trainx)
    aleneg = data[2000:2100]
    alepos = data[2100:]
    aley = [[0,1]] * len(aleneg) + [[1,0]] *len(alepos)
    alex = aleneg + alepos
    aley = np.array(aley)
    alex = np.array(alex)
    #shuffle
    state = np.random.get_state()
    np.random.shuffle(trainx)
    np.random.set_state(state)
    np.random.shuffle(trainy)

    xtest = trainx[-50:]
    ytest = trainy[-50:]
    xtrain = trainx[:-50]
    ytrain = trainy[:-50]
    return xtrain,xtest,ytrain,ytest,alex,aley


def shuffle2():
    data = readvec()
    trainneg = data[:10]
    trainpos = data[10:20]
    trainy = [[0,1]] * len(trainneg) + [[1,0]] *len(trainpos)
    trainy = np.array(trainy)
    trainx = trainneg +trainpos
    trainx = np.array(trainx)
    print(trainx[0][0][0])
    aleneg = data[20:22]
    alepos = data[22:24]
    aley = [[0,1]] * len(aleneg) + [[1,0]] *len(alepos)
    alex = aleneg + alepos
    aley = np.array(aley)
    alex = np.array(alex)
    #shuffle
    state = np.random.get_state()
    np.random.shuffle(trainx)
    np.random.set_state(state)
    np.random.shuffle(trainy)
    print(trainx[0][0][0])
    xtest = trainx[-2:]
    ytest = trainy[-2:]
    xtrain = trainx[:-2]
    ytrain = trainy[:-2]
    print(type(alex),type(xtest))
    return xtrain,xtest,ytrain,ytest,alex,aley

#shuffle2()
