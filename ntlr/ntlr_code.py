import ee
import pandas as pd
import numpy as np
import logging
import sys
import time # For adding a small delay to avoid hammering GEE API

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Earth Engine Project ID ---
# IMPORTANT: Corrected to your actual Google Cloud Project ID from the dashboard screenshot
EE_PROJECT_ID = 'ee-earnestebenezer777' 

# Initialize Earth Engine
try:
    ee.Initialize(project=EE_PROJECT_ID) 
    logger.info(f"Earth Engine initialized successfully for project '{EE_PROJECT_ID}'.")
    
    try:
        current_ee_project = ee.data.get_project()
        logger.info(f"Earth Engine API reports current project as: '{current_ee_project}'")
    except Exception as e:
        logger.error(f"Failed to retrieve current GEE project from API: {e}")

except Exception as e:
    logger.error(f"Failed to initialize Earth Engine. Ensure you have authenticated and have network access. Error: {e}")
    logger.error("Please run 'earthengine authenticate' in your terminal if you haven't.")
    sys.exit(1) # Exit if Earth Engine cannot be initialized

# --- Configuration ---
INPUT_RENAMED_DATA_FILE = 'renamed_dataset.csv'
INPUT_PINCODE_MAP_FILE = 'pincode.csv' # Your pincode to lat/lon mapping file
OUTPUT_RENAMED_WITH_NTLR_FILE = 'renamed_dataset_with_ntlr_first_10.csv' # Changed output filename to reflect subset
PIN_COLUMN = 'pin' # The column in your dataset that holds pincodes (e.g., 'pin' or 'pincode')
NTLR_COLUMN_NAME = 'ntlr' # The new column name for Nightlight data

# === Load Pincode to Lat/Lon Mapping ===
logger.info(f"Loading pincode to Lat/Lon mapping from '{INPUT_PINCODE_MAP_FILE}'...")
try:
    pincode_df = pd.read_csv(INPUT_PINCODE_MAP_FILE)
    pincode_df['pincode'] = pincode_df['pincode'].astype(str)
    
    pincode_df_unique = pincode_df.drop_duplicates(subset=['pincode'], keep='first')
    logger.info(f"Deduplicated pincode mapping. Original: {len(pincode_df)} rows, Unique: {len(pincode_df_unique)} rows.")

    Pincode_LatLon_Map = pincode_df_unique.set_index('pincode')[['latitude', 'longitude']].to_dict('index')
    logger.info(f"Loaded Pincode-Lat/Lon map with {len(Pincode_LatLon_Map)} unique entries.")
except FileNotFoundError:
    logger.error(f"Error: '{INPUT_PINCODE_MAP_FILE}' not found. You MUST provide this file.")
    sys.exit(1)
except KeyError as e:
    logger.error(f"Error: Missing expected column in '{INPUT_PINCODE_MAP_FILE}'. Ensure it has 'pincode', 'latitude', and 'longitude' columns. Error: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred while loading '{INPUT_PINCODE_MAP_FILE}': {e}")
    sys.exit(1)


# === Function to compute Nightlight for a given Lat/Lon ===
def compute_nightlight_from_latlon(lat, lon, start_date='2022-01-01', end_date='2022-12-31', buffer_km=1):
    if pd.isna(lat) or pd.isna(lon) or lat is None or lon is None:
        logger.warning(f"Invalid (NaN/None) lat/lon provided ({lat}, {lon}). Cannot compute Nightlight.")
        return None

    lat = float(lat)
    lon = float(lon)

    point = ee.Geometry.Point(lon, lat)
    buffer_meters = buffer_km * 1000

    image_collection = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
                        .filterBounds(point)
                        .filterDate(start_date, end_date)
                        .select('avg_rad'))

    if image_collection.size().getInfo() == 0:
        logger.warning(f"No VIIRS images found for lat={lat}, lon={lon} in {start_date} to {end_date}. Returning None.")
        return None

    nightlight_image = image_collection.median()

    mean_dict = nightlight_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point.buffer(buffer_meters),
        scale=500,
        maxPixels=1e9
    )

    nightlight_val = mean_dict.getInfo().get('avg_rad')
    return nightlight_val

# === Main Script Execution ===
logger.info(f"Loading renamed dataset from '{INPUT_RENAMED_DATA_FILE}'...")
try:
    # --- THE KEY MODIFICATION ---
    # Load only the first 10 rows of the input CSV
    df_renamed = pd.read_csv(INPUT_RENAMED_DATA_FILE).head(10)
    logger.info(f"Renamed dataset loaded (first {len(df_renamed)} rows only). Shape: {df_renamed.shape}")
except FileNotFoundError:
    logger.error(f"Error: '{INPUT_RENAMED_DATA_FILE}' not found. Please ensure it's in the correct directory.")
    sys.exit(1)

# Ensure the 'pin' column in df_renamed is string type for consistent lookup
df_renamed[PIN_COLUMN] = df_renamed[PIN_COLUMN].astype(str)

# Extract unique pincodes from our *limited* dataset
unique_pincodes_in_data = df_renamed[PIN_COLUMN].unique()
logger.info(f"Found {len(unique_pincodes_in_data)} unique pincodes in the *limited* dataset.")


# Prepare a list of results for Nightlight data
ntlr_data = []
logger.info(f"Fetching Nightlight data for unique pincodes via Earth Engine...")

# Use a dictionary to store fetched NTLR values for unique pincodes
fetched_ntlr_cache = {}

# Iterate only over the unique pincodes found in the *first 10 rows*
for i, pincode in enumerate(unique_pincodes_in_data):
    # Check cache first
    if pincode in fetched_ntlr_cache:
        ntlr_val = fetched_ntlr_cache[pincode]
        ntlr_data.append({'pin': pincode, NTLR_COLUMN_NAME: ntlr_val})
        continue # Skip GEE call if already fetched

    lat_lon_info = Pincode_LatLon_Map.get(pincode)
    
    if lat_lon_info:
        lat = lat_lon_info['latitude']
        lon = lat_lon_info['longitude']
        
        if pd.isna(lat) or pd.isna(lon):
            logger.warning(f"Pincode '{pincode}' has NaN latitude/longitude in mapping file. Skipping Nightlight fetch.")
            ntlr_val = None
        else:
            try:
                ntlr_val = compute_nightlight_from_latlon(lat, lon)
            except Exception as e:
                logger.error(f"Error fetching Nightlight for pincode {pincode} (lat={lat}, lon={lon}): {e}")
                ntlr_val = None
    else:
        logger.warning(f"Pincode '{pincode}' not found in '{INPUT_PINCODE_MAP_FILE}'. Skipping Nightlight fetch for this pincode.")
        ntlr_val = None

    ntlr_data.append({'pin': pincode, NTLR_COLUMN_NAME: ntlr_val})
    fetched_ntlr_cache[pincode] = ntlr_val

    if (i + 1) % 1 == 0 or (i + 1) == len(unique_pincodes_in_data):
        logger.info(f"  Processed {i + 1}/{len(unique_pincodes_in_data)} unique pincodes for Nightlight.")
    
    time.sleep(0.1)

df_ntlr = pd.DataFrame(ntlr_data)
logger.info("Nightlight data fetched and converted to DataFrame.")
print("\nSample of fetched Nightlight data (unique pincodes from first 10 rows):")
print(df_ntlr.head())

# Merge Nightlight data back to the *already limited* df_renamed
df_renamed_with_ntlr = pd.merge(df_renamed, df_ntlr, on=PIN_COLUMN, how='left')

if NTLR_COLUMN_NAME not in df_renamed_with_ntlr.columns:
    logger.error(f"Error: '{NTLR_COLUMN_NAME}' column was not successfully added after merge. Check merge keys and column names.")
    sys.exit(1)
else:
    logger.info(f"Nightlight data merged with renamed dataset (first {len(df_renamed_with_ntlr)} rows). New shape: {df_renamed_with_ntlr.shape}")
    logger.info(f"Sample of renamed dataset with '{NTLR_COLUMN_NAME}' column:")
    print(df_renamed_with_ntlr[[PIN_COLUMN, NTLR_COLUMN_NAME]].head(10)) # Print all 10 rows

# Save the integrated dataset, which now only contains the first 10 rows
df_renamed_with_ntlr.to_csv(OUTPUT_RENAMED_WITH_NTLR_FILE, index=False)
logger.info(f"Integrated renamed dataset (first 10 rows) saved to '{OUTPUT_RENAMED_WITH_NTLR_FILE}'.")
logger.info("--- Nightlight Integration Complete (First 10 Rows Only) ---")