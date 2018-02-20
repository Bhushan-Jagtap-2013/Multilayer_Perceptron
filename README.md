# Multilayer_Perceptron
Neural Network to discriminate between sonar signals bounced off a metal cylinder and those bounced off a roughly cylindrical rock.

## Dataset Location

http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)


## Steps

1. Read dataset
2. One hot encode expected output
3. Shuffle rows
4. Devide data set into Train and Test data
5. Define model with tensor flow and train it on Train data, Repeat for 1000 epoch
6. Test model accuracy on Test data
7. Save model. Also try to restore model and test accuracy for some values from data set

## Data Set Information (from UCI)

The file "sonar.mines" contains 111 patterns obtained by bouncing sonar signals off a metal cylinder at various angles and under various conditions. The file "sonar.rocks" contains 97 patterns obtained from rocks under similar conditions. The transmitted sonar signal is a frequency-modulated chirp, rising in frequency. The data set contains signals obtained from a variety of different aspect angles, spanning 90 degrees for the cylinder and 180 degrees for the rock. 

Each pattern is a set of 60 numbers in the range 0.0 to 1.0. Each number represents the energy within a particular frequency band, integrated over a certain period of time. The integration aperture for higher frequencies occur later in time, since these frequencies are transmitted later during the chirp. 

The label associated with each record contains the letter "R" if the object is a rock and "M" if it is a mine (metal cylinder). The numbers in the labels are in increasing order of aspect angle, but they do not encode the angle directly.
