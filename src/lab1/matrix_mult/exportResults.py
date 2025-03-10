import subprocess
import csv
import matplotlib.pyplot as plt

filenameCSV = 'matrix_mult_results.csv'
filenamePNG = 'matrix_mult_results.png'
filenamePNGall = 'matrix_mult_results_all.png'

# Define the range of values
# start = 1
# end = 64
values = [(i, i, i) for i in [512, 1024, 2048, 4096]]

# Open the CSV file for writing
with open(filenameCSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['hA', 'wA/hB', 'wB', 'Time A (malloc and memcpy)', 'Time B (malloc and memcpy)', 'Time Kernel', 'Time A (malloc and init_matrix)', 'Time B (malloc and init_matrix)', 'Time CPU'])

    # Loop through the range of values
    # for hA in range(start, end + 1):
    #     for wA in range(start, end + 1):
    #         for wB in range(start, end + 1):
    for hA, wA, wB in values:
                # Execute the matrix_mult program with the current parameters
                result = subprocess.run(['./matrix_mult', str(hA), str(wA), str(wB)], capture_output=True, text=True)
                output = result.stdout

                # Parse the output
                # 0 GPU:
                # 1 Time A (malloc and memcpy): X s
                # 2 Time B (malloc and memcpy): X s
                # 3 Time Kernel: X s
                # 4 Time C: X s
                # 5 Bandwidth of A: X KB/s
                # 6 Bandwidth of B: X KB/s
                # 7 Performance Kernel: X GFLOPS/s
                # 8 Bandwidth of C: X KB/s
                # 9 CPU:
                # 10 Time A (malloc and init_matrix): X s
                # 11 Time B (malloc and init_matrix): X s
                # 12 Time CPU: X s
                lines = output.split('\n')
                time_gpu_a = float(lines[1].split(': ')[1].split(' ')[0])
                time_gpu_b = float(lines[2].split(': ')[1].split(' ')[0])
                time_kernel = float(lines[3].split(': ')[1].split(' ')[0])
                time_cpu_a = float(lines[10].split(': ')[1].split(' ')[0])
                time_cpu_b = float(lines[11].split(': ')[1].split(' ')[0])
                time_cpu = float(lines[12].split(': ')[1].split(' ')[0])

                # Write the results to the CSV file
                writer.writerow([hA, wA, wB, time_gpu_a, time_gpu_b, time_kernel, time_cpu_a, time_cpu_b, time_cpu])

# Read the data from the CSV file
data = []
with open(filenameCSV, mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header
    for row in reader:
        data.append([int(row[0]), int(row[1]), int(row[2])] + [float(x) for x in row[3:]])

# Extract the columns
hA = [row[0] for row in data]
wA = [row[1] for row in data]
wB = [row[2] for row in data]
time_gpu_a = [row[3] for row in data]
time_gpu_b = [row[4] for row in data]
time_kernel = [row[5] for row in data]
time_cpu_a = [row[6] for row in data]
time_cpu_b = [row[7] for row in data]
time_cpu = [row[8] for row in data]

# Plot the results
plt.figure(figsize=(12, 8))

# Time Analysis
plt.subplot(2, 2, 1)
plt.plot(range(len(hA)), time_gpu_a, label='Time GPU A (malloc and memcpy)', linestyle='-', marker='o')
plt.plot(range(len(hA)), time_cpu_a, label='Time CPU A (malloc and init_matrix)', linestyle='--', marker='x')
plt.xticks(range(len(hA)), [f'{hA[i]}x{wA[i]}x{wB[i]}' for i in range(len(hA))], rotation=45)
plt.xlabel('Matrix Dimensions')
plt.ylabel('Time (s)')
plt.legend()
plt.title('Time A Analysis')

plt.subplot(2, 2, 2)
plt.plot(range(len(hA)), time_gpu_b, label='Time GPU B (malloc and memcpy)', linestyle='-', marker='o')
plt.plot(range(len(hA)), time_cpu_b, label='Time CPU B (malloc and init_matrix)', linestyle='--', marker='x')
plt.xticks(range(len(hA)), [f'{hA[i]}x{wA[i]}x{wB[i]}' for i in range(len(hA))], rotation=45)
plt.xlabel('Matrix Dimensions')
plt.ylabel('Time (s)')
plt.legend()
plt.title('Time B Analysis')

plt.subplot(2, 2, 3)
plt.plot(range(len(hA)), time_kernel, label='Time Kernel', linestyle='-', marker='o')
plt.plot(range(len(hA)), time_cpu, label='Time CPU', linestyle='--', marker='x')
plt.xticks(range(len(hA)), [f'{hA[i]}x{wA[i]}x{wB[i]}' for i in range(len(hA))], rotation=45)
plt.xlabel('Matrix Dimensions')
plt.ylabel('Time (s)')
plt.legend()
plt.title('Kernel vs CPU Time Analysis')

plt.tight_layout()
plt.savefig(filenamePNG)
plt.show()

# Plot the results
plt.figure(figsize=(12, 8))

# Combined Time Analysis
plt.plot(range(len(hA)), time_gpu_a, label='Time GPU A (malloc and memcpy)', linestyle='-', marker='o')
plt.plot(range(len(hA)), time_cpu_a, label='Time CPU A (malloc and init_matrix)', linestyle='--', marker='x')
plt.plot(range(len(hA)), time_gpu_b, label='Time GPU B (malloc and memcpy)', linestyle='-', marker='s')
plt.plot(range(len(hA)), time_cpu_b, label='Time CPU B (malloc and init_matrix)', linestyle='--', marker='d')
plt.plot(range(len(hA)), time_kernel, label='Time Kernel', linestyle='-', marker='^')
plt.plot(range(len(hA)), time_cpu, label='Time CPU', linestyle='--', marker='v')
plt.xticks(range(len(hA)), [f'{hA[i]}x{wA[i]}x{wB[i]}' for i in range(len(hA))], rotation=45)
plt.xlabel('Matrix Dimensions')
plt.ylabel('Time (s)')
plt.legend()
plt.title('GPU vs CPU Time Analysis')

plt.tight_layout()
plt.savefig(filenamePNGall)
plt.show()