from apt_tele_decode import TelemetryGrabber

# Initialize the grabber with an image and execute  the get channel and visulaize functions
if __name__ == "__main__":
    grabber = TelemetryGrabber("raw2.png")
    grabber.get_channel_type()
    grabber.visualize_telemetry()
