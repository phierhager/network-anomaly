from scapy.all import sniff


def capture_packets(interface="eth0", count=100):
    """Capture packets from a given network interface."""
    packets = sniff(iface=interface, count=count)
    return packets
