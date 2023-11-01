# APT Telemetry Decoder


A decoder for the APT telemetry transmitted alongside the imagery as a bar. 
This decoder takes the raw images from a decoded APT transmission.

It filters the bad telemetry frames out combines all available into one and then can output the instrument channel that 
was transmitted.
The channel association is done based on info found on the wiki of the NOAA 18 satellite and might vary. 

It has been tested with images stemming from a satdump decode. To use with you own images make sure that the telemetry 
bars are not cropped out and no compositing or other manipulation has been done to the image.

A basic example implementation can be found in main. More information on what does what can be found in the 
[wiki](https://github.com/Phoenix-64/APT-Telemetry-Decoder/wiki).

### Update: 
The new `falsecolor_temp()` function calibrates the image values and then shows a falsecolor figure of the temperature. 
The temperature value can be read in the top right corner of the image. To the right image a filter has been applied.

The utilized resources to create this function including all calibration equations are:
* [NOAA user guide](https://noaasis.noaa.gov/NOAASIS/pubs/Users_Guide-Building_Receive_Stations_March_2009.pdf)
* [NOAA KLM user guide](https://www.star.nesdis.noaa.gov/mirs/documents/0.0_NOAA_KLM_Users_Guide.pdf)
## Installation:
Tested with python version 3.10.1 to install the requirements just run:
`pip install -r requirements.txt`

Make sure that you use the same python environment that you installed the requirements to
