import numpy as np
import math


def haversine_distance(p1, p2) -> float:
    """Get distance between 2 world coordinates"""
    lon1, lat1 = p1
    lon2, lat2 = p2

    R = 6371000 # rad of earth

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2 
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance
