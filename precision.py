import numpy as np
import csv
import argparse

from utils import predict

def predict_list(theta0, theta1, input_values):
    return [predict(theta0, theta1, x) for x in input_values]

def r_squared(y_true, y_pred):
    y_mean = sum(y_true) / len(y_true)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_true)
    ss_res = sum((yi - y_hat) ** 2 for yi, y_hat in zip(y_true, y_pred))
    r2 = 1 - (ss_res / ss_tot)
    return r2

def load_thetas(file_path):
    """Load thetas (theta0 and theta1) from a file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        theta0 = float(lines[0].strip())
        theta1 = float(lines[1].strip())
    return theta0, theta1

def load_csv(file_path):
    """Load dataset (input and output) from a CSV file."""
    input = []
    output = []
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            input.append(float(row[0]))
            output.append(float(row[1]))
    
    return np.array(input), np.array(output)

def calculate_thetas_with_numpy(input, output):
    """Use NumPy's polyfit to calculate theta0 and theta1."""
    theta1, theta0 = np.polyfit(input, output, 1)
    return theta0, theta1

def calculate_precision(manual_theta0, manual_theta1, numpy_theta0, numpy_theta1):
    """Calculate the precision (difference) between manual thetas and numpy thetas."""
    precision_theta0 = abs(manual_theta0 - numpy_theta0)
    precision_theta1 = abs(manual_theta1 - numpy_theta1)
    return precision_theta0, precision_theta1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', type=str, default='data.csv', help='Path to the CSV file')
    parser.add_argument('--thetas-path', type=str, default='thetas.txt', help='File where the theta values are stored')
    args = parser.parse_args()

    try:
        manual_theta0, manual_theta1 = load_thetas(args.thetas_path)
    except Exception as e:
        print(f"Error while opening the file {args.thetas_path}")
        print(e)
        exit(1)
    try :
        input, output = load_csv(args.file_path)
    except Exception as e:
        print(f"Error while opening the file {args.file_path}")
        print(e)
        exit(1)
    
    numpy_theta0, numpy_theta1 = calculate_thetas_with_numpy(input, output)
    
    precision_theta0, precision_theta1 = calculate_precision(
        manual_theta0, manual_theta1, numpy_theta0, numpy_theta1
    )
    
    print(f"Manually Calculated Thetas: theta0 = {manual_theta0}, theta1 = {manual_theta1}")
    print(f"NumPy Calculated Thetas:     theta0 = {numpy_theta0}, theta1 = {numpy_theta1}")
    print(f"Thetas Differences:       theta0 = {precision_theta0}, theta1 = {precision_theta1}")

    y_pred_manual = predict_list(manual_theta0, manual_theta1, input)
    y_pred_numpy = predict_list(numpy_theta0, numpy_theta1, input)
    r_squared_manual = r_squared(output, y_pred_manual)
    r_squared_numpy = r_squared(output, y_pred_numpy)

    print(f"R^2 for Manual Thetas: {r_squared_manual}")
    print(f"R^2 for NumPy Thetas:  {r_squared_numpy}")
    print(f"R^2 Difference: {abs(r_squared_manual - r_squared_numpy)}")


if __name__ == '__main__':
    main()
