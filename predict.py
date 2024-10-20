import argparse

import utils

def get_theta_values(file_path):
    try:
        with open(file_path, mode='r') as file:
            theta0 = float(file.readline().strip())
            theta1 = float(file.readline().strip())
    except Exception as e:
        print(f"Error while reading the file {file_path}")
        print(e)
        exit(1)
        
    return theta0, theta1

def predict_output(theta0, theta1):
    try:
        input_value = float(input("Enter the input value: "))
    except Exception as e:
        print("Error while reading the input value")
        print(e)
        exit(1)
        
    output = utils.predict(theta0, theta1, input_value)
    print(f"The output value for the input {input_value} is {output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type=str, default='thetas.txt', help='File with the theta values')
    args = parser.parse_args()
    theta0, theta1 = get_theta_values(args.file_path)
    predict_output(theta0, theta1)