import pystac_client
import planetary_computer
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


print("Connecting to satellite catalog...")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

print("Connected successfully!")


# Our Area of Interest
# [west, south, east, north]
bbox = [80.20, 13.00, 80.30, 13.10]


print("Searching for Sentinel-2 images...")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-01-01/2025-03-31",
    query={"eo:cloud_cover": {"lt": 20}}
)

items = search.item_collection()

print("Images found:", len(items))


# Take the first image
item = items[0]

print("Using scene:")
print(item.id)
print("Date:", item.datetime)


# Sentinel-2 RGB bands
red_url = item.assets["B04"].href
green_url = item.assets["B03"].href
blue_url = item.assets["B02"].href


print("Opening satellite bands...")


with rasterio.open(red_url) as src:

    # Convert our latitude/longitude AOI
    # into the satellite image's coordinate system
    bounds = transform_bounds(
        "EPSG:4326",
        src.crs,
        *bbox
    )

    # Create a small window around our AOI
    window = from_bounds(
        *bounds,
        transform=src.transform
    )

    # Read only the AOI
    red_band = src.read(1, window=window)


with rasterio.open(green_url) as src:

    green_band = src.read(1, window=window)


with rasterio.open(blue_url) as src:

    blue_band = src.read(1, window=window)


print("Satellite data loaded successfully!")


# Stack the three bands
rgb = np.dstack(
    (red_band, green_band, blue_band)
)


# Improve brightness for display
rgb = np.clip(rgb / 3000, 0, 1)


print("Displaying satellite image...")


plt.figure(figsize=(10, 10))

plt.imshow(rgb)

plt.axis("off")

plt.title("Sentinel-2 RGB Image")

plt.show()