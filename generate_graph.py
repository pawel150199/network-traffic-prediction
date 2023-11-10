from visualize_graph import vizualization
from loggers import configureLogger
import argparse
import os

if __name__ == "__main__":
    logger = configureLogger()

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data",
        "-d",
        help="Graph data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_dir",
        "-s",
        help="Where save graph images",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if os.path.exists(args.data):
        path_to_data = args.data
    else:
        logger.error("Path does not exist! Try again.")
        exit(1)

    if not os.path.exists(args.save_dir):
        save_path = args.save_dir
    else:
        logger.warning("File with the same name exists so will be overwrite.")
        save_path = args.save_dir

    try:
        vizualization(path_to_data=path_to_data, save_path=save_path)
        logger.info(f"Successfully saved graph from data in: {os.path.realpath(path_to_data)} to file: {os.path.realpath(save_path)}.")
    except Exception as e:
        logger.error("Graph unsuccessfully generated! Try again.")