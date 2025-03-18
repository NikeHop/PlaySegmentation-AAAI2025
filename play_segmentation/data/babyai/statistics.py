"""
Utilities to compute statistics of the generated datasets
"""

import argparse
import pickle

import blosc
import matplotlib.pyplot as plt


def length_statistics_unsegmented(dataset_file: str) -> None:
    """
    Calculate and visualize statistics for unsegmented trajectory lengths.

    Args:
        dataset_file (str): The path to the dataset file.

    Returns:
        None
    """
    with open(dataset_file, "rb") as f:
        data = pickle.load(f)

    # Prepare statistics
    total_length = 0
    n_traj = 0
    min_length = float("inf")
    max_length = 0
    lengths = []
    unsegmented_lengths = []

    # Collect statistics
    for trajectories in data["images"]:
        unsegmented_length = 0
        for traj in trajectories:
            length = blosc.unpack_array(traj).shape[0]
            lengths.append(length)
            total_length += length
            n_traj += 1
            min_length = min(min_length, length)
            max_length = max(max_length, length)
            unsegmented_length += length
        unsegmented_lengths.append(unsegmented_length)

    print(f"Total number of trajectories: {n_traj}")
    print(f"Avg trajectory length: {total_length/n_traj}")
    print(f"Min trajectory length: {min_length}")
    print(f"Max trajectory length: {max_length}")
    print(f"Max unsegmented length: {max(unsegmented_lengths)}")
    print(f"Min unsegmented length: {min(unsegmented_lengths)}")
    print(
        f"Avg unsegmented length: {sum(unsegmented_lengths)/len(unsegmented_lengths)}"
    )

    plt.clf()
    plt.hist(unsegmented_lengths)
    plt.xlabel("Length of Trajectory")
    plt.ylabel("Frequency")
    plt.savefig("../../visualisations/unsegmented_length_histogram.png")
    plt.savefig("../../visualisations/unsegmented_length_histogram.pdf")

    plt.clf()
    plt.hist(lengths)
    plt.xlabel("Length of Trajectory")
    plt.ylabel("Frequency")
    plt.savefig("../../visualisations/length_histogram.png")
    plt.savefig("../../visualisations/length_histogram.pdf")


def length_statistics_single(dataset_file: str) -> None:
    """
    Compute and print statistics about the lengths of trajectories in a dataset.

    Args:
        dataset_file (str): The path to the dataset file.

    Returns:
        None
    """
    # Load data
    with open(dataset_file, "rb") as f:
        data = pickle.load(f)

    # Prepare statistics
    total_length = 0
    n_traj = 0
    min_length = float("inf")
    max_length = 0
    lengths = []

    # Compute statistics
    for traj in data["images"]:
        length = blosc.unpack_array(traj).shape[0]
        lengths.append(length)
        total_length += length
        n_traj += 1
        min_length = min(min_length, length)
        max_length = max(max_length, length)

    # Print statistics
    print(f"Total number of trajectories: {n_traj}")
    if n_traj > 0:
        print(f"Avg trajectory length: {total_length/n_traj}")
    print(f"Min trajectory length: {min_length}")
    print(f"Max trajectory length: {max_length}")

    plt.hist(lengths)
    plt.xlabel("Length of Trajectory")
    plt.ylabel("Frequency")
    plt.savefig("../../visualisations/length_histogram.png")
    plt.savefig("../../visualisations/length_histogram.pdf")


def get_statistics(dataset_file: str, single: bool) -> None:
    """
    Calculate statistics for a given dataset file.

    Parameters:
        dataset_file (str): The path to the dataset file.
        single (bool): If True, calculate statistics for a single segment. If False, calculate statistics for the entire dataset.

    Returns:
        None
    """
    if single:
        length_statistics_single(dataset_file)
    else:
        length_statistics_unsegmented(dataset_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="./default.pkl",
        help="The dataset file to compute statistics over",
    )
    parser.add_argument(
        "--single", action="store_true", default=False, help="Single segment dataset"
    )
    args = parser.parse_args()

    get_statistics(args.file, single=args.single)
