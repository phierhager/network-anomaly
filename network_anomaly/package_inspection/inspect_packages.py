from network_anomaly.package_inspection.packet_capture import capture_packets
from network_anomaly.package_inspection.feature_extraction import (
    extract_features_real_time,
)
from network_anomaly.package_inspection.random_forest_model import (
    classify_packet,
)


def inspect_real_time(interface="eth0", count=10):
    packets = capture_packets(interface, count)
    for packet in packets:
        features = extract_features_real_time(packet)
        classification = classify_packet(features)
        print(f"Packet classified as: {classification}")


if __name__ == "__main__":
    inspect_real_time(interface="wlo1", count=10)
