import os

import foxglove
from foxglove import Channel

from utils import open_csv


def generate_weather_viz(folder, topic_name="/weather") -> None:
    print(f"Weather data from: {folder} with topic {topic_name}")

    channel = Channel(topic=topic_name)

    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        data, fields = open_csv(filepath)
        for i, data in enumerate(data):
            stamp = int(data["Stamp"])
            weather_data = {}
            for field in fields:
                if field in ["Sequence"]:
                    continue
                weather_data[field] = data[field]

            channel.log(
                weather_data,
                log_time=stamp,
            )

            print(f"Processed data {i + 1}/{len(data)}: {file}")


if __name__ == "__main__":
    writer = foxglove.open_mcap("weather.mcap", allow_overwrite=True)
    generate_weather_viz(
        "/Volumes/SandiskSSD/JLMV/Foxglove/REHEARSE RAIN/data/000/weather")
