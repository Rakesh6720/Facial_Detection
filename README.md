# Facial Detection
A Python Convolutional Neural Network Trained for Facial Recognition

Convolutional Neural Networks (CNN) learn to recognize faces in images by training on a dataset which contains not only images of people but also a table of keypoints indexed by key-value pairs.  For example, the dataset on which this notebook trained indexed facial keypoints in a CSV catalogued by (x,y) positions.  Each image in the dataset included 68 coordinate pairs, which formed a keypoint mask of the face in an image.

The 3 notebooks in this repository demonstrate the model's success at identifying faces of different qualitative variety including gender, race, and ethnicity.  

The Python file in the repository shows the architecture of the CNN which, through the PyTorch API, implements convolutional layers for feature detection, Max Pooling layers for dimensionality reduction, dropout layers to prevent overfitting, and rectlinear functions for regularization.

The model trained using a Mean Squared loss function and the Adam optimizer for gradient descent.  

The model implemented batch training where batch size was 32 for 50 epochs.  
