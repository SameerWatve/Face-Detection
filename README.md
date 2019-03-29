# Face-Detection

Objectives: Face image classification using Gaussian model, Mixture of Gaussian model, tdistribution, Mixture of t-distribution, Factor Analysis 

Data processing:

FDDB data set is used for the project.

Initially as per project description, images were resized to 60*60 and all three RGB channels were taken into account. Each image was forming set of 60*60*3 = 10800 set of features. While calculating Gaussian Probability Distribution the issue of overflow was encountered in Python.

Hence images have been downscaled to the point where the problem of overflow is solved. 10*10 was the maximum possible resolution. It is observed that converting images to gray scale reduces run time of program significantly. Hence now we only have 10*10*1 = 100 set of features for every image which are easy to handle by the program.

The diagonal co-variance matrix is considered in this project for all the models. It is assumed that each pixel is independent of every other pixel and hence its co-variance matrix has only diagonal elements.
