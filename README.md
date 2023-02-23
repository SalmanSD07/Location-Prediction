# Location-Prediction
Predicting location based on MLEnd sound dataset
LEndClassification
Simple ML Classification Task based on sound data

Basic Solution
Intonation Prediction

In this project I have built a model that predicts the intonation of a short audio segment. The audio dataset was prepared by the batch of MSc Big Data Science at QMUL.

The MLEND.csv consits of 20k rows and 6 columns. Each row corresponds to one of the items in our dataset, and each item is described by four attributes.

File ID (audio file)
Area
spot
in_out
Participand ID
Intonation
Here I have extracted features from the audio signal namely and classified weather the audio was indoor or outdoor.

Power
Pitch mean
Pitch std
Voiced flag
Onset
Intensity
Short time Fourier transform(STFT)
Spectral Centroid
Spectral Bandwidth
Spectral Contrast
Spectral Flatness
Spectral Rolloff
Root Mean Square
using librosa library.

I build 3 models which are

SVM
RandomForestClassifier
DecisionTreeClassifier

RandomForestClassifier is used to find the best parameters of each of the models. The models take the features as input and try to classify them into the the 4 intonations.

Finally, Getting the highest Accuracy on the Random Forest Classifier model with the accuracy of 61.9% on the validation data with normalised predictors.

Advanced Solution
Next, I decided to develop a model that given audio features and area predicts the spot within that area.

The Approach For this multiclass classification problem was to take features from audio and pass the index of area and predict index of spot.

All audio files are iterated through and the features are extracted as well as the 6 label classes.
The numpy array values are saved on the drive
NaN values are removed
The numpy arrays are converted to a dataframe for easy of encoding and preprocessing
The dataframe is transformed using a Standard Scaler
PCA is performed to extract 2 and 4 sets of vectors.
Data Visualized to analyse class distribution, correlation and relations between features.
Data is split between training and testing set
Again out of all models svm, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier the RandomForestClassifier works the best and gave the validation accuracy of 31.95%. Confusion Matrix is displayed at the end to demonstrate the result.

