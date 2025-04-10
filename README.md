###CRC
def xor(a, b):
    return ''.join(['0' if i == j else '1' for i, j in zip(a, b)])
def mod2div(dividend, divisor):
    pick = len(divisor)
    tmp = dividend[:pick]
    while pick < len(dividend):
        tmp = xor(divisor, tmp) + dividend[pick] if tmp[0] == '1' else xor('0'*pick, tmp) + dividend[pick]
        pick += 1
    tmp = xor(divisor, tmp) if tmp[0] == '1' else xor('0'*pick, tmp)
    return tmp
def encode_crc(data, key):
    padded = data + '0' * (len(key)-1)
    remainder = mod2div(padded, key)
    return data + remainder
def detect_crc_error(received, key):
    remainder = mod2div(received, key)
    return "Error Detected" if '1' in remainder else "No Error"
data = input("Enter data bits: ")
key = input("Enter generator polynomial bits: ")
transmitted = encode_crc(data, key)
print("Transmitted frame:", transmitted)
bit_pos = int(input("Enter bit position to flip (starting from 0): "))
errored = list(transmitted)
errored[bit_pos] = '0' if errored[bit_pos] == '1' else '1'
errored = ''.join(errored)
print("Received with error:", errored)
print("CRC Check Result:", detect_crc_error(errored, key))

##TCP text
##Server
import socket
filename = input("Enter filename to save received file: ")
server = socket.socket()
server.bind(('localhost', 9001))
server.listen(1)
print("Server listening...")
conn, addr = server.accept()
print("Connected to:", addr)
with open(filename, "w") as f:
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        f.write(data)
print("File received.")
conn.close()

##Client
import socket
client = socket.socket()
client.connect(('localhost', 9001))
file_to_send = input("Enter text file name to send: ")
with open(file_to_send, "r") as f:
    data = f.read(1024)
    while data:
        client.send(data.encode())
        data = f.read(1024)
print("File sent.")
client.close()

#username password
import socket
credentials = {"admin": "admin123", "user": "user123"}
server = socket.socket()
server.bind(('localhost', 9020))
server.listen(1)
print("Server ready for login...")
conn, addr = server.accept()
data = conn.recv(1024).decode()
username, password = data.split(',')
msg = "Authenticated" if credentials.get(username) == password else "Failed"
conn.send(msg.encode())
conn.close()

import socket
client = socket.socket()
client.connect(('localhost', 9020))
username = input("Enter username: ")
password = input("Enter password: ")
client.send(f"{username},{password}".encode())
response = client.recv(1024).decode()
print("Server Response:", response)
client.close()

#UDP
import socket
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.bind(('localhost', 7001))
print("UDP Echo Server ready...")
while True:
    data, addr = server.recvfrom(1024)
    print("Received:", data.decode())
    server.sendto(data, addr)

import socket
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
msg = input("Enter message for Echo UDP server: ")
client.sendto(msg.encode(), ('localhost', 7001))
data, _ = client.recvfrom(1024)
print("Echoed back:", data.decode())
##chat
import socket
import threading
def receive_messages(conn):
    while True:
        msg = conn.recv(1024).decode()
        if not msg or msg.lower() == 'exit':
            print("Client has left the chat.")
            break
        print(f"Client: {msg}")
def chat_server():
    server = socket.socket()
    server.bind(('localhost', 9050))
    server.listen(1)
    print("Server is waiting for a connection...")
    conn, addr = server.accept()
    print(f"Connected to {addr}")
    threading.Thread(target=receive_messages, args=(conn,)).start()
    while True:
        msg = input("You: ")
        conn.send(msg.encode())
        if msg.lower() == 'exit':
            break
    conn.close()
    server.close()
if __name__ == "__main__":
    chat_server()

import socket
import threading
def receive_messages(client):
    while True:
        msg = client.recv(1024).decode()
        if not msg or msg.lower() == 'exit':
            print("Server has left the chat.")
            break
        print(f"Server: {msg}")
def chat_client():
    client = socket.socket()
    client.connect(('localhost', 9050))
    print("Connected to server.")
    threading.Thread(target=receive_messages, args=(client,)).start()
    while True:
        msg = input("You: ")
        client.send(msg.encode())
        if msg.lower() == 'exit':
            break
    client.close()
if __name__ == "__main__":
    chat_client()

##parity
import socket
def calculate_parity(data, mode='even'):
    ones = data.count('1')
    if mode == 'even':
        return '0' if ones % 2 == 0 else '1'
    else:
        return '1' if ones % 2 == 0 else '0'
def check_parity(data_with_parity, mode='even'):
    data = data_with_parity[:-1]
    received_parity = data_with_parity[-1]
    expected_parity = calculate_parity(data, mode)
    return received_parity == expected_parity
def start_server():
    s = socket.socket()
    s.bind(('localhost', 9001))
    s.listen(1)
    print("Server listening on port 9001...")
    conn, addr = s.accept()
    print("Connected by", addr)
    data = conn.recv(1024).decode()
    mode = conn.recv(1024).decode()
    print("Received data with parity:", data)
    print("Using parity mode:", mode)
    if check_parity(data, mode):
        result = "No Error Detected"
    else:
        result = "Error Detected"
    conn.send(result.encode())
    conn.close()
    s.close()
if __name__ == "__main__":
    start_server()

import socket
def calculate_parity(data, mode='even'):
    ones = data.count('1')
    if mode == 'even':
        return '0' if ones % 2 == 0 else '1'
    else:
        return '1' if ones % 2 == 0 else '0'
def add_parity_bit(data, mode='even'):
    parity = calculate_parity(data, mode)
    return data + parity
def start_client():
    s = socket.socket()
    s.connect(('localhost', 9001))
    data = input("Enter binary data: ")
    mode = input("Enter parity mode (even/odd): ").strip().lower()
    frame = add_parity_bit(data, mode)
    print("Sending data with parity:", frame)
    # Optional: Flip a bit to simulate error
    flip = input("Flip a bit? (y/n): ").strip().lower()
    if flip == 'y':
        pos = int(input("Enter bit position to flip (0-based): "))
        frame = list(frame)
        frame[pos] = '0' if frame[pos] == '1' else '1'
        frame = ''.join(frame)
        print("Modified data (simulated error):", frame)
    s.send(frame.encode())
    s.send(mode.encode())
    result = s.recv(1024).decode()
    print("Server response:", result)
    s.close()
if __name__ == "__main__":
    start_client()

##symmetric
# aes_server.py
from cryptography.fernet import Fernet
import socket
key = Fernet.generate_key()
cipher = Fernet(key)
server = socket.socket()
server.bind(('localhost', 9010))
server.listen(1)
print("Server Ready")
conn, addr = server.accept()
print("Connected to:", addr)
conn.send(key)  # Send symmetric key
encrypted_data = conn.recv(1024)
decrypted = cipher.decrypt(encrypted_data).decode()
print("Decrypted:", decrypted)
conn.close()

# aes_client.py
from cryptography.fernet import Fernet
import socket
client = socket.socket()
client.connect(('localhost', 9010))
key = client.recv(1024)
cipher = Fernet(key)
msg = "Secret Message using AES"
encrypted = cipher.encrypt(msg.encode())
client.send(encrypted)
client.close()

##linkstate
import heapq
def dijkstra(graph, start):
    shortest = {node: float('inf') for node in graph}
    shortest[start] = 0
    visited = set()
    heap = [(0, start)]
    while heap:
        (dist, current) = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        for neighbor, cost in graph[current].items():
            if dist + cost < shortest[neighbor]:
                shortest[neighbor] = dist + cost
                heapq.heappush(heap, (shortest[neighbor], neighbor))
    return shortest
# Example graph
graph = {
    'A': {'B': 2, 'C': 5},
    'B': {'A': 2, 'C': 1, 'D': 4},
    'C': {'A': 5, 'B': 1, 'D': 2},
    'D': {'B': 4, 'C': 2}
}
start = input("Enter start node (e.g., A): ")
shortest_paths = dijkstra(graph, start)
print("Shortest paths from", start)
for dest, cost in shortest_paths.items():
    print(f"{start} -> {dest}: {cost}")

##distance vector
def distance_vector_routing(graph):
    nodes = list(graph.keys())
    dist = {node: {n: float('inf') for n in nodes} for node in nodes}
    for node in nodes:
        dist[node][node] = 0
        for neighbor in graph[node]:
            dist[node][neighbor] = graph[node][neighbor]
    # Bellman-Ford style updates
    for _ in range(len(nodes) - 1):
        for node in nodes:
            for neighbor in graph[node]:
                for dest in nodes:
                    if dist[node][dest] > dist[node][neighbor] + dist[neighbor][dest]:
                        dist[node][dest] = dist[node][neighbor] + dist[neighbor][dest]
    return dist
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 7},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 7, 'C': 1}
}
routing_table = distance_vector_routing(graph)
for node in routing_table:
    print(f"\nRouting table for {node}")
    for dest in routing_table[node]:
        print(f"{node} -> {dest}: {routing_table[node][dest]}")

#ospf
def ospf_simulation(graph, source):
    print(f"=== OSPF Simulation from Node {source} ===")
    paths = dijkstra(graph, source)
    for dest, cost in paths.items():
        print(f"{source} -> {dest} | Cost: {cost}")
    print()
def dijkstra(graph, source):
    visited = {node: False for node in graph}
    distance = {node: float('inf') for node in graph}
    distance[source] = 0
    pq = [(0, source)]
    while pq:
        dist, current = heapq.heappop(pq)
        if visited[current]:
            continue
        visited[current] = True
        for neighbor, cost in graph[current].items():
            if distance[current] + cost < distance[neighbor]:
                distance[neighbor] = distance[current] + cost
                heapq.heappush(pq, (distance[neighbor], neighbor))
    return distance
# Network topology
network = {
    'R1': {'R2': 1, 'R3': 4},
    'R2': {'R1': 1, 'R3': 2, 'R4': 5},
    'R3': {'R1': 4, 'R2': 2, 'R4': 1},
    'R4': {'R2': 5, 'R3': 1}
}
for router in network:
    ospf_simulation(network, router)

##dec to bin
def ip_to_binary(ip_address):
    # Split the IP into octets
    octets = ip_address.strip().split(".")
    # Check if IP is valid
    if len(octets) != 4 or not all(o.isdigit() and 0 <= int(o) <= 255 for o in octets):
        return "Invalid IP address format."
    # Convert each octet to binary and pad with leading zeros
    binary_octets = [format(int(octet), '08b') for octet in octets]
    # Join the binary octets with dots
    binary_ip = ".".join(binary_octets)
    return binary_ip
# Get user input
user_ip = input("Enter an IP address: ")
binary_representation = ip_to_binary(user_ip)
print("Binary representation:", binary_representation)

##bin to dec
def binary_to_ip(binary_ip):
    # Split the binary string into 4 octets
    octets = binary_ip.strip().split(".")
    # Validate that there are exactly 4 octets and each has 8 bits
    if len(octets) != 4 or not all(len(octet) == 8 and set(octet) <= {'0', '1'} for octet in octets):
        return "Invalid binary IP address format."
    # Convert each 8-bit binary string to decimal
    decimal_octets = [str(int(octet, 2)) for octet in octets]
    # Join decimal values into dotted format
    return ".".join(decimal_octets)
# Get user input
binary_input = input("Enter a binary IP address: ")
decimal_ip = binary_to_ip(binary_input)
# Display result
print("Dotted decimal format:", decimal_ip)

##go back n
import random
import time
def go_back_n_send(frames, window_size):
    base = 0
    while base < len(frames):
        end = min(base + window_size, len(frames))
        print(f"\nSender: Sending frames {base} to {end - 1}")
        for i in range(base, end):
            print(f"Sent frame: {frames[i]}")
        # Simulate ACKs (random loss simulation)
        acks_received = True
        for i in range(base, end):
            ack_lost = random.choice([True, False])  # Simulate random ACK loss
            if ack_lost:
                print(f"ACK for frame {frames[i]} lost! Retransmitting from frame {frames[i]}")
                acks_received = False
                break
            else:
                print(f"ACK received for frame {frames[i]}")
        if acks_received:
            base = end
        else:
            time.sleep(1)  # Simulate timeout
# === Run the simulation ===
frames = [f"F{i}" for i in range(1, 11)]  # 10 frames
window_size = int(input("Enter window size for Go-Back-N: "))
go_back_n_send(frames, window_size)

#selective repeat
import random
import time
def selective_repeat_send(frames, window_size):
    base = 0
    received = [False] * len(frames)
    while not all(received):
        for i in range(base, min(base + window_size, len(frames))):
            if not received[i]:
                print(f"Sender: Sending frame {frames[i]}")
                # Simulate success or loss
                if random.choice([True, False]):
                    print(f"Receiver: Received frame {frames[i]}, sending ACK")
                    received[i] = True
                else:
                    print(f"Receiver: Frame {frames[i]} lost! No ACK")

        # Slide window
        while base < len(frames) and received[base]:
            base += 1
        time.sleep(1)  # Simulate delay
    print("\nAll frames transmitted successfully with Selective Repeat.")
# === Run the simulation ===
frames = [f"F{i}" for i in range(1, 11)]  # 10 frames
window_size = int(input("Enter window size for Selective Repeat: "))
selective_repeat_send(frames, window_size)

##checksum
def binary_addition(a, b):
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)
    result = ''
    carry = 0
    for i in range(max_len - 1, -1, -1):
        total = carry
        total += int(a[i]) + int(b[i])
        result = str(total % 2) + result
        carry = total // 2
    if carry:
        result = '1' + result
    return result[-max_len:] if len(result) > max_len else result
def ones_complement(binary_str):
    return ''.join('1' if b == '0' else '0' for b in binary_str)
def calculate_checksum(data_chunks):
    sum_result = data_chunks[0]
    for chunk in data_chunks[1:]:
        sum_result = binary_addition(sum_result, chunk)
        # If overflow, wrap-around carry
        if len(sum_result) > len(chunk):
            sum_result = binary_addition(sum_result[1:], '1')
    checksum = ones_complement(sum_result)
    return checksum
def verify_checksum(data_chunks, checksum):
    sum_result = checksum
    for chunk in data_chunks:
        sum_result = binary_addition(sum_result, chunk)
        if len(sum_result) > len(chunk):
            sum_result = binary_addition(sum_result[1:], '1')
    return all(b == '1' for b in sum_result)
# === Main Simulation ===
data = input("Enter binary data (space-separated 8-bit blocks): ")
chunks = data.strip().split()
# Sender side
checksum = calculate_checksum(chunks)
print("Calculated Checksum:", checksum)
# Simulate sending and receiving
received = chunks.copy()
received_checksum = checksum
# Receiver side
valid = verify_checksum(received, received_checksum)
if valid:
    print(" No error detected.")
else:
    print(" Error detected in received data.")

##subnetting
import ipaddress
import math
def generate_subnets(network, desired_subnets):
    # Calculate how many bits are needed to achieve at least the desired number of subnets.
    additional_bits = math.ceil(math.log(desired_subnets, 2))
    new_prefix = network.prefixlen + additional_bits
    
    if new_prefix > 32:
        raise ValueError(f"Cannot create {desired_subnets} subnets from {network} (not enough bits available).")   
    # Generate all subnets using the new prefix length.
    subnets = list(network.subnets(new_prefix=new_prefix))
    return subnets
def main():
    # Get the network input from the user in CIDR notation.
    network_input = input("Enter the network :")
    try:
        network = ipaddress.ip_network(network_input, strict=False)
    except ValueError as e:
        print("Invalid network input:", e)
        return
    # Get the desired number of subnets from the user.
    desired_subnets_input = input("Enter the number of subnets you want to create: ")
    try:
        desired_subnets = int(desired_subnets_input)
        if desired_subnets < 1:
            print("Number of subnets must be a positive integer.")
            return
    except ValueError:
        print("Invalid input. Please enter a numeric value for the number of subnets.")
        return
    # Generate the subnets.
    try:
        subnets = generate_subnets(network, desired_subnets)
    except ValueError as e:
        print(e)
        return
    print("\nGenerated Subnets:")
    # If more subnets are generated than requested, display only the first 'desired_subnets' subnets.
    for i, subnet in enumerate(subnets[:desired_subnets], start=1):
        print(f"Subnet {i}: {subnet}")
if __name__ == "__main__":
    main()

