from apt_tele_decode import TelemetryGrabber

# Initialize the grabber with an image and execute  the get channel and visulaize functions
if __name__ == "__main__":
    grabber = TelemetryGrabber("noaa18_2c.png", 1)
    tel = grabber.generate_telemetry()
    grabber.get_channel_type()
    grabber.visualize_telemetry()
    grabber.falsecolor_temp(bars=False)
    #grabber.visualize_telemetry()

