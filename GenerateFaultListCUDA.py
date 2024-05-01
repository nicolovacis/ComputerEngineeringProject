import numpy as np
import struct
import math
import random
import csv
import argparse

def save_tensors(tensors, filename):
    # Open a file in binary write mode
    with open(filename, 'wb') as f:
        # Write the number of tensors
        num_tensors = len(tensors)
        f.write(np.int32(num_tensors))

        # Write each tensor's shape and data
        for tensor in tensors:
            # Ensure the tensor is a 4D numpy array
            if tensor.ndim != 4:
                raise ValueError("Each tensor must be a 4D numpy array.")

            # Write the shape of the tensor
            f.write(np.array(tensor.shape, dtype=np.int32).tobytes())
            contiguous_tensor = np.ascontiguousarray(tensor, dtype=np.float32)
            # Write the tensor data
            f.write(contiguous_tensor.astype(np.float32).tobytes())

def generateFaultList(file_input, file_output, inj_number):

    integers = []

    # Open a binary input file
    with open(file_input, 'rb') as file:

        # Read number of tensor
        num_tensor = struct.unpack('<i', file.read(4))[0]

        # Number of inj for each tensor (approximated to the upper case)
        single_inj_num = math.ceil(inj_number/num_tensor)

        # Open csv output file
        with open(file_output, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(["inj_id"] + ["tensor_id"] + ["n"] + ["c"] + ["h"] + ["w"] + ["bit_pos"])

            global_injection_counter = 0

            for it_tensor_id in range(num_tensor):

                # Read the 4 ints which defines the shape of the tensor
                n_dim, c_dim, h_dim, w_dim = struct.unpack('<iiii', file.read(16))
                total_data = n_dim * c_dim * h_dim * w_dim

                # Read the float data
                data = []
                for _ in range(total_data):
                    data_value = struct.unpack('<f', file.read(4))[0]
                    data.append(data_value)

                # Calculate the injection only on the weight tensor (2nd position)
                if it_tensor_id%2!=0:

                    tensor_id = (it_tensor_id-1)//2

                    # Repeat single_inj_num injections
                    for inj_id in range(single_inj_num):

                        global_injection_counter += 1

                        n = random.randint(0, n_dim - 1)
                        c = random.randint(0, c_dim - 1)
                        h = random.randint(0, h_dim - 1)
                        w = random.randint(0, w_dim - 1)

                        bit_pos = random.randint(0, 31) # data (float) is expressed by 32 bits

                        # Writing a new row in the csv file
                        spamwriter.writerow([global_injection_counter] + [tensor_id] + [n] + [c] + [h] + [w] + [bit_pos])


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='GenerateFaultListCUDA',
        description='Script that generates a fault list for CUDA',
    )

    parser.add_argument('--injections_number', type=int, required=True)
    parser.add_argument('--bin_file_tensors', type=str, required=True)
    parser.add_argument('--csv_file_output', type=str, required=True)

    args = parser.parse_args()

    injections_number = args.injections_number
    bin_file_tensors = args.bin_file_tensors
    csv_file_output = args.csv_file_output

    # Create some example tensors with different shapes
    tensors = [
        np.random.rand(32, 64, 32, 32).astype(np.float32),
        np.random.rand(128, 64, 3, 3).astype(np.float32),
        np.random.rand(16, 256, 7, 7).astype(np.float32),
        np.random.rand(128, 256, 3, 3).astype(np.float32),
        np.random.rand(4, 3, 416, 416).astype(np.float32),
        np.random.rand(32, 3, 3, 3).astype(np.float32),
    ]

    # Save to file
    save_tensors(tensors, bin_file_tensors)
    generateFaultList(bin_file_tensors, csv_file_output, injections_number)