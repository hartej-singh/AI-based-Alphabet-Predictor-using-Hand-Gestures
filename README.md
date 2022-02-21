# AI-based-Letter-Predictor-using-Hand-Gestures

This is a CNN model (Convolution Neural Network) that takes in a static image as input of a hand gesture and gives an output of the predicted letter.

To run, execute the AI model code and generate a .h5 file. Copy-paste that model-save file in the Flask folder.

Dataset used: https://www.kaggle.com/ash2703/handsignimages
Dataset details: There are a total of 27,455 gray-scale images of size 28*28 pixels whose value range between 0-255. Each case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions).
