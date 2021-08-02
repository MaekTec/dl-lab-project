import numpy as np
from pprint import pprint
import os
import sys

sys.path.insert(0, os.getcwd())
import argparse
import csv
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_files', nargs='*', help="csv files from tensorboard")
    parser.add_argument('--name', type=str, help="name of file and title")
    parser.add_argument('--output-root', type=str, default='plots')
    parser.add_argument('--xlabel', type=str, default="", help="xlabel")
    parser.add_argument('--ylabel', type=str, default="", help="ylabel")
    parser.add_argument('--labels', nargs='*', help="labels")
    args = parser.parse_args()

    return args


def main(args):
    small = True
    show = False
    save = True

    x_values_list = []
    metric_values_list = []
    for csv_filename in args.csv_files:
        x_values = []
        metric_values = []
        with open(csv_filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for i, row in enumerate(csv_reader):
                x_values.append(float(row["Step"]))
                metric_values.append(float(row["Value"]))
        x_values_list.append(x_values)
        metric_values_list.append(metric_values)

    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})  # no cut offs
    if small:
        from matplotlib.pyplot import figure
        figure(num=None, figsize=(6.4, 4.8), dpi=100)  # default: figure(num=None, figsize=(6.4, 4.8), dpi=100)

    for i, (x_value, metric_value) in enumerate(zip(x_values_list, metric_values_list)):
        if args.labels is None:
            plt.plot(x_value, metric_value)
        else:
            plt.plot(x_value, metric_value, label=args.labels[i].replace("_", " "))
    plt.xlabel(args.xlabel.replace("_", " "))
    plt.ylabel(args.ylabel.replace("_", " "))
    # plt.title(args.name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if show:
        plt.show()
    if save:
        plt.savefig(os.path.join(args.output_root, "{}.pdf".format(args.name)))


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
