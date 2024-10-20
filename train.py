import csv
import argparse
import math
import matplotlib.pyplot as plt
import os

from utils import predict

GRAPH_FOLDER = './graphs'

tmp_theta0_list = []
tmp_theta1_list = []

def display_graph(data):
    plt.scatter(data['original_input'], data['output'])
    plt.xlabel(data['names'][0])
    plt.ylabel(data['names'][1])
    plt.savefig(f"{GRAPH_FOLDER}/graph.png")
    #plt.show()
    plt.clf()

def display_graph_with_line(data, theta0, theta1):
    plt.scatter(data['original_input'], data['output'])
    plt.xlabel(data['names'][0])
    plt.ylabel(data['names'][1])
    x_values = [min(data['original_input']), max(data['original_input'])]
    y_values = [predict(theta0, theta1, x) for x in x_values]
    plt.plot(x_values, y_values, color='red')
    plt.savefig(f"{GRAPH_FOLDER}/graph_with_line.png")
    #plt.show()
    plt.clf()

def display_tmp_theta_graph():
    plt.scatter(range(1, len(tmp_theta0_list) + 1), tmp_theta0_list)
    plt.xlabel('Iteration')
    plt.ylabel('tmp_theta0')
    plt.savefig(f"{GRAPH_FOLDER}/tmp_theta0.png")
    #plt.show()
    plt.clf()

    plt.scatter(range(1, len(tmp_theta1_list) + 1), tmp_theta1_list)
    plt.xlabel('Iteration')
    plt.ylabel('tmp_theta1')
    plt.savefig(f"{GRAPH_FOLDER}/tmp_theta1.png")
    #plt.show()
    plt.clf()


def calculate_standard_deviation(data):
    if len(data) == 0:
        return []
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = math.sqrt(variance)
    return std_dev

def calculate_mean(data):
    if len(data) == 0:
        return []
    return sum(data) / len(data)

def scale_data_compute(data, mean, std_dev):
    if len(data) == 0:
        return []
    return [(x - mean) / std_dev for x in data]

def scale_data(data):
    mean = calculate_mean(data)
    std_dev = calculate_standard_deviation(data)
    return scale_data_compute(data, mean, std_dev)

def unscale_theta0(theta0, theta1, mean, std_dev):
    return theta0 - theta1 * mean / std_dev

def unscale_theta1(theta1, std_dev):
    return theta1 / std_dev

def unscale_thetas(theta0, theta1, input):
    mean = calculate_mean(input)
    std_dev = calculate_standard_deviation(input)
    theta0 = unscale_theta0(theta0, theta1, mean, std_dev)
    theta1 = unscale_theta1(theta1, std_dev)
    return theta0, theta1

def can_create_file(file_path):
    try:
        with open(file_path, mode='w') as file:
            pass
    except Exception as e:
        print(f"Error while creating the file {file_path}")
        print(e)
        exit(1)

def file_exists(file_path):
    try:
        with open(file_path, mode='r') as file:
            pass
    except Exception as e:
        print(f"Error while opening the file {file_path}")
        print(e)
        exit(1)


def load_csv(file_path):
    data = {
        'names': [],
        'input': [],
        'output': []
    }
    
    with open(file_path, mode='r') as file:
        try:
            reader = csv.reader(file)
            data['names'] = next(reader)
            for row in reader:
                data['input'].append(float(row[0]))
                data['output'].append(float(row[1]))
        except Exception as e:
            print("Error while reading the file")
            print(e)
            exit(1)
            
    return data

def calculate_tmp_theta0(learning_rate, input, output, theta0, theta1):
    m = len(input)
    summation = sum((predict(theta0, theta1, input[i]) - output[i]) for i in range(m))
    tmp_theta0 = learning_rate * (1 / m) * summation
    tmp_theta0_list.append(tmp_theta0)
    return tmp_theta0

def calculate_tmp_theta1(learning_rate, input, output, theta0, theta1):
    m = len(input)
    summation = sum((predict(theta0, theta1, input[i]) - output[i]) * input[i] for i in range(m))
    tmp_theta1 = learning_rate * (1 / m) * summation
    tmp_theta1_list.append(tmp_theta1)
    return tmp_theta1

def train(data, iterations, precision):
    theta0 = 0
    theta1 = 0
    for i in range(iterations):
        tmp_theta0 = calculate_tmp_theta0(data['learning_rate'], data['input'], data['output'], theta0, theta1)
        tmp_theta1 = calculate_tmp_theta1(data['learning_rate'], data['input'], data['output'], theta0, theta1)
        if abs(tmp_theta0) < precision and abs(tmp_theta1) < precision:
            break
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
    return theta0, theta1

def store_thetas(theta0, theta1, thetas_path):
    with open(thetas_path, mode='w') as file:
        file.write(f"{theta0}\n")
        file.write(f"{theta1}\n")

def create_folder(folder):
    try:
        if os.path.exists(folder) and os.path.isdir(folder):
            return
        os.makedirs(folder)
    except Exception as e:
        print(f"Error while creating the folder {folder}")
        print(e)
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Linear regression tool")
    parser.add_argument('--file-path', type=str, default='data.csv', help='Path to the CSV file')
    parser.add_argument('--learning-rate', type=float, default=1, help='Learning rate for the model')
    parser.add_argument('--iterations', type=int, default=100000, help='Number of iterations for the training')
    parser.add_argument('--thetas-path', type=str, default='thetas.txt', help='File where the theta values will be stored')
    parser.add_argument('--graph-folder', type=str, default='./graphs', help='Folder where the graphs will be stored')
    parser.add_argument('--precision', type=float, default=0.000001, help='Precision for the training')
    args = parser.parse_args()

    if args.iterations <= 0:
        print("Number of iterations must be positive")
        exit(1)

    file_exists(args.file_path)
    can_create_file(args.thetas_path)
    GRAPH_FOLDER = args.graph_folder
    create_folder(GRAPH_FOLDER)

    data = load_csv(args.file_path)
    data['original_input'] = data['input']
    display_graph(data)

    data['input'] = scale_data(data['original_input'])
    data['learning_rate'] = args.learning_rate
    
    theta0, theta1 = train(data, args.iterations, args.precision)
    theta0, theta1 = unscale_thetas(theta0, theta1, data['original_input'])
    store_thetas(theta0, theta1, args.thetas_path)
    print(f"theta0: {theta0}, theta1: {theta1}")

    display_graph_with_line(data, theta0, theta1)
    display_tmp_theta_graph()


    




