from apt_tele_decode import TelemetryGrabber

# Initialize the grabber with an image and execute  the get channel and visulaize functions
if __name__ == "__main__":
    grabber = TelemetryGrabber("raw1.png", 0)
    tel = grabber.generate_telemetry()
    grabber.get_channel_type()
    grabber.visualize_telemetry()
    grabber.falsecolor_temp(bars=True)
    #grabber.visualize_telemetry()

