"# research_paper_2018" 

By Franz Lomibao

This Contains two Juypter Notebook written in Pycharm

Rnn_classifier is the classifier that I made and it classifies gestures

Run From top and Install all required document
    
    Tensorflow 1.9.0

This document works by taking the 6D data of gestures and Splits the data by 
    
    Trial and User
    
Combine All feature columns 
    
    posX, posY, posZ ...
    
Pad features to 
    
    300 rows and N ammount of feature Colmuns
   
Create the classification of the gestures

    0: Circle Horizontal Counter Clockwise Gesture
    1: V Shape Gesture
    2: X Shape Gesture
    3: Twist Counter Clockwise Gesture
    4: Circle Vertical Counter Clockwise Gesture
    
Create all the input Function
    
    Training_input_fn
    test_input_fn
    predict_input_fn

Slice the feature/train input data into'
    
    300 by number of columns
    
Create the hidden layers

Create the Estimators

Train and Evaluate

Results

Predictions 