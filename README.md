# 542-Final-Project
enviroment needed:
  1: python > 3.6
  2: tensorflow
  3: IPython
  4: bert-serving-server (only for the bert-lstmcnn model) - see: https://pypi.org/project/bert-serving-server/
  5: Keras(only for keras-lstm model)

 
 
There are 3 folders:

1: keras-lstm: the baseline model, it has problem due to gradient vanishing
        Can be use when we have an input file, but the outcome can be bad when the text is not good.
        If you need to change the input text , you need to change the directory in the getvector.py file.
        Then we can call:
         
              python train_keras.py

2: lstmcnn: the normal lstm-cnn approach.
        we can just use the command below to run the model and the result will be in a log.txt file.
        if you want to try the other data, need to change the file directory in getvector.py.
        
              python lstmcnn.py
        
        
3: bert-lstmcnn:
        the bert-lstm-cnn approach:
        We use the smalldata.csv to generate our sentence vector since the bert vectors are way too large(more than 100g).
        However, the sentence vector of smalldata dataset is still larger than 10g, so we can not upload it here.
        We run this model on the scc server.
        We can run getvector_bert to transform the input sentences into bert vectors(sentence embedding).
        and use readvector to read it.
        in the end, we can run the model by using the command: 
              
              python bert-lstmcnn.py
              
       
        
