import subprocess
import csv
import matplotlib.pyplot as plt

filenameCSV = 'matrix_mult_results.csv'
filenamePNG = 'matrix_mult_results.png'

# Define the range of values
# start = 1
# end = 64
values = [
    (1, 1, 1),
    (3, 1, 3),
    (16, 16, 16),
    (16, 17, 16),
    (32, 32, 32)
]

# Open the CSV file for writing
with open(filenameCSV, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['hA', 'wA/hB', 'wB', 'Time A (malloc and memcpy)', 'Time B (malloc and memcpy)', 'Time Kernel', 'Time C (cudaMemcpyDeviceToHost)', 'Band width of A', 'Band width of B', 'Performance Kernel', 'Band width of C'])

    # Loop through the range of values
    # for hA in range(start, end + 1):
    #     for wA in range(start, end + 1):
    #         for wB in range(start, end + 1):
    for hA, wA, wB in values:
                # Execute the matrix_mult program with the current parameters
                result = subprocess.run(['./matrix_mult', str(hA), str(wA), str(wB)], capture_output=True, text=True)
                output = result.stdout

                # Parse the output
                lines = output.split('\n')
                time_a = float(lines[0].split(': ')[1].split(' ')[0])
                time_b = float(lines[1].split(': ')[1].split(' ')[0])
                time_kernel = float(lines[2].split(': ')[1].split(' ')[0])
                time_c = float(lines[3].split(': ')[1].split(' ')[0])
                bandwidth_a = float(lines[4].split(': ')[1].split(' ')[0])
                bandwidth_b = float(lines[5].split(': ')[1].split(' ')[0])
                performance_kernel = float(lines[6].split(': ')[1].split(' ')[0])
                bandwidth_c = float(lines[7].split(': ')[1].split(' ')[0])

                # Write the results to the CSV file
                writer.writerow([hA, wA, wB, time_a, time_b, time_kernel, time_c, bandwidth_a, bandwidth_b, performance_kernel, bandwidth_c])

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
time_a = [row[3] for row in data]
time_b = [row[4] for row in data]
time_kernel = [row[5] for row in data]
time_c = [row[6] for row in data]
bandwidth_a = [row[7] for row in data]
bandwidth_b = [row[8] for row in data]
performance_kernel = [row[9] for row in data]
bandwidth_c = [row[10] for row in data]

# Plot the results
plt.figure(figsize=(12, 8))

# Time Analysis
plt.subplot(2, 2, 1)
plt.plot(range(len(hA)), time_a, label='Time A (malloc and memcpy)')
plt.plot(range(len(hA)), time_b, label='Time B (malloc and memcpy)')
plt.plot(range(len(hA)), time_kernel, label='Time Kernel')
plt.plot(range(len(hA)), time_c, label='Time C (cudaMemcpyDeviceToHost)')
plt.xticks(range(len(hA)), [f'{hA[i]}x{wA[i]}x{wB[i]}' for i in range(len(hA))], rotation=45)
plt.xlabel('Matrix Dimensions')
plt.ylabel('Time (s)')
plt.legend()
plt.title('Time Analysis')

# Bandwidth Analysis
plt.subplot(2, 2, 2)
plt.plot(range(len(hA)), bandwidth_a, label='Bandwidth A')
plt.plot(range(len(hA)), bandwidth_b, label='Bandwidth B')
plt.plot(range(len(hA)), bandwidth_c, label='Bandwidth C')
plt.xticks(range(len(hA)), [f'{hA[i]}x{wA[i]}x{wB[i]}' for i in range(len(hA))], rotation=45)
plt.xlabel('Matrix Dimensions')
plt.ylabel('Bandwidth (KB/s)')
plt.legend()
plt.title('Bandwidth Analysis')

# Performance Analysis
plt.subplot(2, 2, 3)
plt.plot(range(len(hA)), performance_kernel, label='Performance Kernel')
plt.xticks(range(len(hA)), [f'{hA[i]}x{wA[i]}x{wB[i]}' for i in range(len(hA))], rotation=45)
plt.xlabel('Matrix Dimensions')
plt.ylabel('Performance (GFLOPS/s)')
plt.legend()
plt.title('Performance Analysis')

plt.tight_layout()
plt.savefig(filenamePNG)
plt.show()