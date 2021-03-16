# Image-Panorama-with-openCV
A Panorama implementation for stitching together images. The implementation is in C++ and stitches together 4 images. 
For the extraction of the features, two different common algorithms were tested:
* The SIFT feature detections algorithm as well as 
* The SURF feature detection algorithm. 

The images are stitched together using the mathcing keypoints. No greyscale correction algorithms are applied on the output image, thus some greyscale differences are visible. 
