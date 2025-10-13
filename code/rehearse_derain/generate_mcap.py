import os

from foxglove import open_mcap
from process_radar_csv import generate_radar_viz
from process_images import generate_image_viz
from process_lidar import generate_lidar_viz
from process_weather import generate_weather_viz


ROOT_FOLDER = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data", "rehearse_derain")

SEQUENCES = ["051"]

OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, "output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def generate_mcap():
    mcap_filename = "rehearse_derain.mcap"
    mcap_path = os.path.join(OUTPUT_FOLDER, mcap_filename)
    writer = open_mcap(mcap_path, allow_overwrite=True)

    for sequence in SEQUENCES:
        print(f"Processing sequence: {sequence}")
        FLIR_IMGS_FOLDER = os.path.join(DATA_FOLDER, sequence, "flir")
        ARENA_IMGS_FOLDER = os.path.join(
            DATA_FOLDER, sequence, "arena")
        RADAR_FOLDER = os.path.join(DATA_FOLDER, sequence, "ext_radar")
        INNOVIZ_FOLDER = os.path.join(DATA_FOLDER, sequence, "innoviz")
        OUSTER_FOLDER = os.path.join(DATA_FOLDER, sequence, "ouster")
        WEATHER_FOLDER = os.path.join(DATA_FOLDER, sequence, "weather")

        generate_image_viz(FLIR_IMGS_FOLDER, f"{sequence}/flir")
        generate_image_viz(ARENA_IMGS_FOLDER, f"{sequence}/arena")
        generate_radar_viz(RADAR_FOLDER, f"{sequence}/radar")
        generate_lidar_viz(INNOVIZ_FOLDER, f"{sequence}/innoviz")
        generate_lidar_viz(OUSTER_FOLDER, f"{sequence}/ouster")
        generate_weather_viz(WEATHER_FOLDER, f"{sequence}/weather")

    writer.close()

    print(f"MCAP file generated at: {mcap_path}")


if __name__ == "__main__":
    generate_mcap()
