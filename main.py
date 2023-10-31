from apttelemetrydecoder import TelemetryGrabber

if __name__ == "__main__":
    grabber = TelemetryGrabber("raw2.png")
    grabber.get_channel_type()
    grabber.visualize_telemetry()
