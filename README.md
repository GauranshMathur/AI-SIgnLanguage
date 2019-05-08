# AI-SignLanguage
Application that converts ASL into english using tensorflow keras and computer vision

## Using the demo files. 
To use the demo files you need to have python installed with keras and tensorflow packages. Most of it all can be downloaded through Anaconda.

To run the program you can either open it in spyder and run it directly from there or go to compand promt and type
```
python demo.py
```
This is for the static letters

```
python dynamic-cam.py
```
This is for the letters J and Z

## Systems used
The model used was a pre-created model found online. But the filters were changed as to fit the image sizing. You can look at the model under the folder pyimagesearch.

## Datasets
The datasets created used 2 different filters to be trained to get the model to comprehend all 24 static letters in the alphabet. Because a lot of the alphabets are quite similar to each other.

The two types of filters used were:
* A background masking filter
* A hand counturing filter
