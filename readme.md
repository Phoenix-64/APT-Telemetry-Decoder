# APT Telemetry Decoder


A decoder for the APT telemetry transmitted alongside the imagery as a bar. 
This decoder takes the raw images from a decoded APT transmission.

It filters the bad telemetry frames out combines all available into one and then can output the instrument channel that 
was transmitted.
The channel association is done based on info found on the wiki of the NOAA 18 satellite and might vary. 

It has been tested with images stemming from a satdump decode. To use with you own images make sure that the telemetry 
bars are not cropped out and no compositing or other manipulation has been done to the image.

A basic example implementation can be found in main.

In the future one could also add output color value calibration based on the instrument calibration strips 
to get accurate temperature data. But so far I could not find the necessary info online.

## Installation:
Tested with python version 3.10.1 to install the requirements just run:
`pip install -r requirements.txt`

Make sure that you use the same python environment that you installed the requirements to
