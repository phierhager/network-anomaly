from scapy.all import *
from scapy.all import IP, TCP, UDP, ICMP, Raw
from collections import defaultdict
from time import time
import numpy as np

# Global session tracking
session_stats = defaultdict(
    lambda: {
        "spkts": 0,
        "dpkts": 0,
        "sbytes": 0,
        "dbytes": 0,
        "sinpkt": [],
        "dinpkt": [],
    }
)
last_packet_time = defaultdict(lambda: 0)


def calculate_duration(packet):
    if packet.haslayer("IP"):
        src = packet["IP"].src
        now = time()
        if session_stats[src]["spkts"] == 0:  # First packet
            session_stats[src]["start_time"] = now
        session_stats[src]["end_time"] = now
        return session_stats[src]["end_time"] - session_stats[src]["start_time"]
    return 0


def map_protocol(packet):
    if packet.haslayer(TCP):
        return 1  # TCP
    elif packet.haslayer(UDP):
        return 2  # UDP
    elif packet.haslayer(ICMP):
        return 3  # ICMP
    return 0  # Other


def map_service(packet):
    if packet.haslayer(TCP) and packet.haslayer(Raw):
        payload = packet[Raw].load.decode(errors="ignore")
        if "HTTP" in payload:
            return 1  # HTTP
        elif "FTP" in payload:
            return 2  # FTP
    return 0  # Unknown


def map_state(packet):
    if packet.haslayer(TCP):
        flags = packet[TCP].flags
        if flags == "S":
            return 1  # SYN
        if flags == "SA":
            return 2  # SYN-ACK
        if flags == "FA":
            return 3  # FIN-ACK
    return 0  # Other


def count_source_packets(packet):
    if packet.haslayer(IP):
        src = packet[IP].src
        session_stats[src]["spkts"] += 1
        return session_stats[src]["spkts"]
    return 0


def count_dest_packets(packet):
    if packet.haslayer(IP):
        dst = packet[IP].dst
        session_stats[dst]["dpkts"] += 1
        return session_stats[dst]["dpkts"]
    return 0


def calculate_source_bytes(packet):
    if packet.haslayer(IP):
        src = packet[IP].src
        length = len(packet)
        session_stats[src]["sbytes"] += length
        return session_stats[src]["sbytes"]
    return 0


def calculate_dest_bytes(packet):
    if packet.haslayer(IP):
        dst = packet[IP].dst
        length = len(packet)
        session_stats[dst]["dbytes"] += length
        return session_stats[dst]["dbytes"]
    return 0


def calculate_rate(packet):
    duration = calculate_duration(packet)
    if duration > 0:
        return (
            calculate_source_bytes(packet) + calculate_dest_bytes(packet)
        ) / duration
    return 0


def calculate_source_load(packet):
    return calculate_source_bytes(packet) / max(1, count_source_packets(packet))


def calculate_dest_load(packet):
    return calculate_dest_bytes(packet) / max(1, count_dest_packets(packet))


def calculate_source_loss(packet):
    return max(
        0, session_stats[packet[IP].src]["spkts"] - count_dest_packets(packet)
    )


def calculate_dest_loss(packet):
    return max(
        0, session_stats[packet[IP].dst]["dpkts"] - count_source_packets(packet)
    )


def calculate_source_interval(packet):
    src = packet[IP].src
    now = time()
    interval = now - last_packet_time[src]
    last_packet_time[src] = now
    session_stats[src]["sinpkt"].append(interval)
    return interval


def calculate_dest_interval(packet):
    dst = packet[IP].dst
    now = time()
    interval = now - last_packet_time[dst]
    last_packet_time[dst] = now
    session_stats[dst]["dinpkt"].append(interval)
    return interval


def calculate_source_jitter(packet):
    src = packet[IP].src
    intervals = session_stats[src]["sinpkt"]
    if len(intervals) < 2:
        return 0
    return np.std(intervals)


def calculate_dest_jitter(packet):
    dst = packet[IP].dst
    intervals = session_stats[dst]["dinpkt"]
    if len(intervals) < 2:
        return 0
    return np.std(intervals)


def extract_source_window(packet):
    return packet[TCP].window if packet.haslayer(TCP) else 0


def calculate_rtt(packet):
    if packet.haslayer(TCP):
        return packet[TCP].ack - packet[TCP].seq
    return 0


def calculate_transaction_depth(packet):
    if packet.haslayer(TCP) and packet.haslayer(Raw):
        payload = packet[Raw].load.decode(errors="ignore")
        if "GET" in payload or "POST" in payload:
            return 1
    return 0


def count_source_ports(session_stats):
    return len(
        {
            p.sport
            for p in session_stats.keys()
            if (p.haslayer(TCP) or p.haslayer(UDP)) and hasattr(p, "sport")
        }
    )


def count_dest_ports(packet):
    return len(
        {
            p.dport
            for p in session_stats.keys()
            if p.haslayer(TCP) or p.haslayer(UDP)
        }
    )


def extract_source_tcp_base(packet):
    if packet.haslayer(TCP):
        return packet[
            TCP
        ].seq  # TCP Sequence number (can be used as base for source)
    return 0


def extract_dest_tcp_base(packet):
    if packet.haslayer(TCP):
        return packet[
            TCP
        ].ack  # TCP Acknowledgment number (can be used as base for destination)
    return 0


def extract_dest_window(packet):
    if packet.haslayer(TCP):
        return packet[TCP].window  # TCP Window size
    return 0


syn_time = {}
ack_time = {}


def calculate_synack_delay(packet):
    if packet.haslayer(TCP):
        if packet[TCP].flags == "S":  # SYN packet
            syn_time[packet[IP].src] = time()  # Record SYN time
        elif packet[TCP].flags == "SA":  # SYN-ACK packet
            if packet[IP].dst in syn_time:  # If SYN was recorded
                delay = time() - syn_time[packet[IP].dst]
                return delay
    return 0


def calculate_ack_delay(packet):
    if packet.haslayer(TCP):
        if packet[TCP].flags == "A":  # ACK packet
            if packet[IP].dst in ack_time:
                return time() - ack_time[packet[IP].dst]
            ack_time[packet[IP].dst] = time()  # Record ACK time
    return 0


source_size = defaultdict(list)


def calculate_source_mean(packet):
    if packet.haslayer(IP):
        src = packet[IP].src
        source_size[src].append(
            len(packet)
        )  # Store packet sizes for the source
        return np.mean(source_size[src])  # Mean packet size for the source
    return 0


dest_size = defaultdict(list)


def calculate_dest_mean(packet):
    if packet.haslayer(IP):
        dst = packet[IP].dst
        dest_size[dst].append(
            len(packet)
        )  # Store packet sizes for the destination
        return np.mean(dest_size[dst])  # Mean packet size for the destination
    return 0


def calculate_response_body_length(packet):
    if packet.haslayer(Raw):
        payload = packet[Raw].load.decode(errors="ignore")
        if "HTTP" in payload and "Content-Length" in payload:
            # Extract Content-Length header from HTTP response
            content_length_index = payload.find("Content-Length")
            content_length = payload[content_length_index:].split("\r\n")[0]
            try:
                return int(
                    content_length.split(":")[1].strip()
                )  # Return the length of the response body
            except ValueError:
                return 0
    return 0


def check_ftp_login(packet):
    if packet.haslayer(Raw):
        payload = packet[Raw].load.decode(errors="ignore")
        if "USER" in payload or "PASS" in payload:
            return True  # FTP login detected (either USER or PASS command)
    return False


ftp_command_count = defaultdict(int)


def count_ftp_commands(packet):
    if packet.haslayer(Raw):
        payload = packet[Raw].load.decode(errors="ignore")
        if "USER" in payload or "PASS" in payload or "QUIT" in payload:
            src = packet[IP].src
            ftp_command_count[src] += 1  # Increment count for this source IP
            return ftp_command_count[src]
    return 0


http_method_count = defaultdict(int)


def count_http_methods(packet):
    if packet.haslayer(Raw):
        payload = packet[Raw].load.decode(errors="ignore")
        if (
            "GET" in payload
            or "POST" in payload
            or "PUT" in payload
            or "DELETE" in payload
        ):
            src = packet[IP].src
            http_method_count[src] += 1  # Increment count for this source IP
            return http_method_count[src]
    return 0


ip_port_pair_count = defaultdict(int)


def check_same_ip_ports(packet):
    if packet.haslayer(TCP) or packet.haslayer(UDP):
        src_dst_pair = (
            packet[IP].src,
            packet[IP].dst,
            packet.sport,
            packet.dport,
        )
        ip_port_pair_count[src_dst_pair] += 1
        if (
            ip_port_pair_count[src_dst_pair] > 5
        ):  # Example threshold: more than 5 packets to the same IP-port pair
            return True
    return False
