import os
import csv


def open_csv(file_path: str) -> list[dict]:
    """
        Open a CSV file and return its contents as a list of dictionaries
        Input:
            file_path (str): Path to the CSV file
        Output:
            list[dict]: List of dictionaries representing the CSV rows

    """
    with open(file_path, "r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        fields = csv_reader.fieldnames
        data = list(csv_reader)
    # print(f"Available data fields: {fields}")
    return data, fields


def get_metadata_from_folder(folder) -> list[dict]:
    """
    Reads metadata from a folder containing CSV files.
    Args:
        folder (str): Path to the folder containing CSV files.
    Returns:
        list[dict]: List of dictionaries containing metadata from each CSV file.
    """
    metadata = []
    metadata_path = os.path.join(folder, "metadata.csv")
    if os.path.exists(metadata_path):
        print(f"Metadata file found in {folder}")
        metadata, _ = open_csv(metadata_path)

    return metadata


def generate_palette(value, num_colors=256):
    # Ensure the value is between 0 and 1
    value = max(0, min(1, value))

    # Generate a palette from red to green
    palette = []
    for i in range(num_colors):
        # Calculate the RGB values
        red = int(255 * (1 - i / (num_colors - 1)))
        green = int(255 * (i / (num_colors - 1)))
        blue = 0
        palette.append((red, green, blue))

    # Select a color based on the input value
    selected_index = int(value * (num_colors - 1))
    selected_color = palette[selected_index]
    r, g, b = selected_color[0], selected_color[1], selected_color[2]

    return r, g, b
