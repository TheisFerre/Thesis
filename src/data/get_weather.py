from datetime import datetime
from meteostat import Hourly, Point


def get_weather_info(start: datetime, stop: datetime, point: Point):

    start1 = datetime(2018, 12, 1)

    end1 = datetime(2018, 12, 31)

    data = Hourly(point, start=start1, end=end1)

    data = data.normalize()
    data = data.aggregate("4H")
    data = data.fetch()
    return data[["temp", "rhum", "prcp", "wspd"]]


def get_nearby_point(longitude: float, lattitude: float):

    point = Point(lon=longitude, lat=lattitude)

    return point


if __name__ == '__main__':
    point = get_nearby_point(longitude=-74.002776, lattitude=40.760875)

    data = get_weather_info(0, 0, point)

    print(data)


