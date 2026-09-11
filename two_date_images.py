import pystac_client
import planetary_computer
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.windows import from_bounds


print("Connecting to satellite catalog...")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

print("Connected successfully!")


# -----------------------------------------
# AREA OF INTEREST
# [west, south, east, north]
# -----------------------------------------

bbox = [80.20, 13.00, 80.30, 13.10]


# -----------------------------------------
# FUNCTION TO GET ONE SATELLITE IMAGE
# -----------------------------------------

def get_satellite_image(date_range):

    print("\nSearching for:", date_range)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}}
    )

    items = search.item_collection()

    if len(items) == 0:
        print("No images found!")
        return None

    # Select first available scene
    item = items[0]

    print("Selected scene:")
    print(item.id)
    print("Date:", item.datetime)

    # -----------------------------------------
    # GET SENTINEL-2 BANDS
    # -----------------------------------------

    red_url = item.assets["B04"].href
    green_url = item.assets["B03"].href
    blue_url = item.assets["B02"].href
    scl_url = item.assets["SCL"].href


    # -----------------------------------------
    # READ RED BAND
    # -----------------------------------------

    with rasterio.open(red_url) as src:

        # Convert our latitude/longitude AOI
        # into the satellite image coordinate system

        bounds = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox
        )

        # Create a window containing only our AOI

        window = from_bounds(
            *bounds,
            transform=src.transform
        )

        # Read only the AOI

        red = src.read(1, window=window)

        # Save information about the RGB grid

        rgb_transform = src.window_transform(window)
        rgb_crs = src.crs


    # -----------------------------------------
    # READ GREEN BAND
    # -----------------------------------------

    with rasterio.open(green_url) as src:

        green = src.read(1, window=window)


    # -----------------------------------------
    # READ BLUE BAND
    # -----------------------------------------

    with rasterio.open(blue_url) as src:

        blue = src.read(1, window=window)


    # -----------------------------------------
    # READ SCL CLOUD CLASSIFICATION
    # -----------------------------------------

    with rasterio.open(scl_url) as src:

        # Convert AOI to SCL coordinate system

        scl_bounds = transform_bounds(
            "EPSG:4326",
            src.crs,
            *bbox
        )

        scl_window = from_bounds(
            *scl_bounds,
            transform=src.transform
        )

        scl = src.read(1, window=scl_window)

        scl_transform = src.window_transform(scl_window)
        scl_crs = src.crs


    # -----------------------------------------
    # RESAMPLE SCL TO RGB RESOLUTION
    # -----------------------------------------

    scl_resampled = np.zeros(
        red.shape,
        dtype=np.uint8
    )

    reproject(
        source=scl,
        destination=scl_resampled,

        src_transform=scl_transform,
        src_crs=scl_crs,

        dst_transform=rgb_transform,
        dst_crs=rgb_crs,

        resampling=Resampling.nearest
    )


    # -----------------------------------------
    # CREATE CLOUD MASK
    # -----------------------------------------

    cloud_mask = (
        (scl_resampled == 3) |
        (scl_resampled == 8) |
        (scl_resampled == 9) |
        (scl_resampled == 10)
    )


    # -----------------------------------------
    # CREATE RGB IMAGE
    # -----------------------------------------

    rgb = np.dstack(
        (red, green, blue)
    )


    # -----------------------------------------
    # HIDE CLOUD PIXELS
    # -----------------------------------------

    rgb = np.moveaxis(rgb, 0, -1)

    alpha = np.ones(rgb.shape[:2])
    alpha[cloud_mask] = 0

    rgb = np.dstack((rgb, alpha))


    # -----------------------------------------
    # BRIGHTNESS ADJUSTMENT
    # -----------------------------------------

    rgb = np.clip(
        rgb / 3000,
        0,
        1
    )


    print("Image loaded successfully!")

    return rgb


# =========================================
# GET BEFORE IMAGE
# =========================================

print("\nGetting BEFORE image...")

before = get_satellite_image(
    "2025-01-01/2025-01-31"
)


# =========================================
# GET AFTER IMAGE
# =========================================

print("\nGetting AFTER image...")

after = get_satellite_image(
    "2025-03-01/2025-03-31"
)


# =========================================
# DISPLAY BOTH IMAGES
# =========================================

if before is not None and after is not None:

    print("\nBoth images loaded successfully!")

    plt.figure(figsize=(12, 6))


    # BEFORE IMAGE

    plt.subplot(1, 2, 1)

    plt.imshow(before)

    plt.title(
        "BEFORE - January 2025"
    )

    plt.axis("off")


    # AFTER IMAGE

    plt.subplot(1, 2, 2)

    plt.imshow(after)

    plt.title(
        "AFTER - March 2025"
    )

    plt.axis("off")


    plt.tight_layout()

    plt.show()