import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from botocore import UNSIGNED
from botocore.config import Config
import os
import pvlib

'''
AE 498 Fall 2026
Data reduction code for solar degradation analysis project
WARNING: This code takes several minutes to run. 
It is recommended to download the csv output from GitHub in order
to run solar_degradation.py. 
However, if you want to run data_reduction, follow the instructions
in the README.md
'''

# Configuration

# link to primary metadata: https://openei.org/wiki/PVDAQ/Sites/NREL_x-Si_6

BUCKET = "oedi-data-lake"
LONGITUDE = -105.1727
LATITUDE = 39.742 
TIME_STEP = 60 # minutes (don't change!)
TEST_THERMAL_CONST = -0.5/100 # 1/(degree C)
G_STC =  1000 # W/m2 standard test irradiance
P_STC = 6140 # W AC https://nlr.gov/docs/legosti/old/21285.pdf
SYSTEM_ID = 50 # NREL x-Si 6 in Golden, CO, around 39.742, -105.1727 
TILT = 45 # degrees from horizontal
AZIMUTH = 158 # pvlib convention: degrees clockwise from north (180 = south)
AMBIENT_TEMP_MEAN = 32.65 # degrees C (average cell temp 2011-2022 using pvdaq data)

START_YEAR = 2001
END_YEAR = 2023
SKIP_YEAR = 2010 # no data folder available in 2010 for system 50
API_KEY = os.environ.get("NSRDB_API_KEY", "DEMO_KEY")  # DEMO_KEY is heavily rate limited
#API_KEY = 'your_api_key'
EMAIL = os.environ.get("NSRDB_EMAIL", "your_email")

# Get NSRDB Data Note: this data extraction is subject to rate limits,
# which can produce fatal errors. If the code outputs an HTTP error,
# wait a moment and try again.

def get_local_poa_irradiance(year):
    """
    Pull NSRDB psm4 aggregated data (for a specific year) for a location and transpose it to plane-of-array
    (POA) irradiance for a fixed-tilt surface.
    """
    # map_variables=True renames columns to pvlib names (ghi, dni, dhi, temp_air,
    # wind_speed, albedo, pressure, ...).
    data, meta = pvlib.iotools.get_nsrdb_psm4_aggregated(
        LATITUDE, LONGITUDE,
        api_key=API_KEY, email=EMAIL,
        year=year,
        time_step=TIME_STEP,
        map_variables=True,
        leap_day=True,
    )

    print("Site metadata:", {k: meta.get(k) for k in ("Location ID", "Time Zone", "Elevation")})

    # Solar position

    altitude = meta.get("altitude", meta.get("Elevation", 0))

    solpos = pvlib.solarposition.get_solarposition(
        data.index,
        LATITUDE,
        LONGITUDE,
        altitude=altitude,
        pressure=data["pressure"] * 100 if "pressure" in data else None,  # mbar -> Pa
        temperature=data["temp_air"] if "temp_air" in data else 12,
    )

    # Transposition
    
    # Extraterrestrial DNI is needed by the Perez (and Hay-Davies) sky diffuse models.
    dni_extra = pvlib.irradiance.get_extra_radiation(data.index)

    # Relative airmass is needed by the Perez model.
    airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

    albedo = data["albedo"] if "albedo" in data else 0.2  # fall back to a typical value

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=TILT,
        surface_azimuth=AZIMUTH,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=data["dni"], # direct normal irradiance
        ghi=data["ghi"], # global horizontal irradiance
        dhi=data["dhi"], # direct horizontal irradiance
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=albedo,
        model="perez", # alternatives: 'isotropic', 'haydavies', 'klucher', 'reindl'
        model_perez="allsitescomposite1990",
    )
    # poa columns: poa_global, poa_direct, poa_diffuse, poa_sky_diffuse, poa_ground_diffuse

    # Results
    
    #result = pd.concat([data[["ghi", "dni", "dhi"]], poa], axis=1)
    result = poa["poa_global"]
    
    return result


# Get PVDAQ Data ---> daily power, temperature (per minute)

# Select columns to extract from CSV files

COLUMNS_TO_READ = [
    "measured_on",
    "ac_power__752",
    "module_temp_1__760",
    "module_temp_2__761"
]

# Anonymous S3 connection
s3 = boto3.client(
    "s3",
    region_name="us-west-2",
    config=Config(signature_version=UNSIGNED)
)

# Find all available files from pvdaq (all valid years)
# Retrieve corresponding irradiance

def get_pvdaq_files(system_id, start_year, end_year):

    files = []
    irradiances = []

    paginator = s3.get_paginator("list_objects_v2")

    for year in range(start_year, end_year + 1):  
        if year == SKIP_YEAR:
            continue
        prefix = (
            f"pvdaq/csv/pvdata/"
            f"system_id={system_id}/"
            f"year={year}/"
        )

        print(f"Searching {prefix}")

        for page in paginator.paginate(
            Bucket=BUCKET,
            Prefix=prefix
        ):

            for obj in page.get("Contents", []):

                key = obj["Key"]

                if key.endswith(".csv"):
                    files.append(key)
        irradiance = get_local_poa_irradiance(year)
        irradiances.append(irradiance)
            
    print('irradiances shape',len(irradiances))
    return files, irradiances



# Extract date information from S3 key
def get_date_from_key(key):

    parts = key.split("/")

    # initialize
    year = None
    month = None
    day = None

    for part in parts:

        if part.startswith("year="):
            year = int(part.split("=")[1])

        elif part.startswith("month="):
            month = int(part.split("=")[1])

        elif part.startswith("day="):
            day = int(part.split("=")[1])

    return year, month, day

# Process file ---> power, temperature, irradiance for 1 day (hourly)
def process_file(key, irradiance):

    year, month, day = get_date_from_key(key)

    print(f"Processing {year}-{month:02d}-{day:02d}")

    response = s3.get_object(
        Bucket=BUCKET,
        Key=key
    )

    # Only load the columns specified above.
    df = pd.read_csv(
        BytesIO(response["Body"].read()),
        usecols=COLUMNS_TO_READ
    ) 
    #print("df\n\n",df.head(10))
    #print(df[df["measured_on"]=="measured_on"])
    # daily irradiance:
    if START_YEAR < SKIP_YEAR < END_YEAR: # case where SKIP_YEAR is involved
        if year > SKIP_YEAR: 
            irradiance = irradiance[year-START_YEAR-1].loc[f"{year}-{month}-{day}"] 
        else: 
            irradiance = irradiance[year-START_YEAR].loc[f"{year}-{month}-{day}"] 
    else:
        irradiance = irradiance[year-START_YEAR].loc[f"{year}-{month}-{day}"] 
    # Resample pvdaq data at intervals = TIME_STEP
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    df = df.set_index("measured_on")
    df = df.resample(f"{TIME_STEP}min").mean()
    
    # both are in local time (make timezone-blind)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    irradiance.index = pd.to_datetime(irradiance.index).tz_localize(None)
    
    # insert irradiance column into df
    df = pd.merge_asof(
        df.sort_index(),
        irradiance.sort_index().rename("poa_irradiance"),
        left_index=True,
        right_index=True,
        direction="nearest"
    ) 
    
    # Drop all rows with negative power generation
    df = df[df[f"ac_power__752"] > 0]

    #print("module_temp_2__761 *****\n",s,"\n******")

    # Check whether enough data remain
    if len(df) == 0:
        print("    NO VALID DATA")
        return {
                "year": year,
                "month": month,
                "key": key, # result is still hourly
                "EN_ac": np.nan, # hourly energy generation
                "EN_stc_corrected": np.nan, # W-h weather-corrected estimate of power output 
                "T_cell_avg": np.nan, # W-h weather-corrected estimate of power output 
                "poa_irradiance":np.nan, # W/m2
            }

    print(f"    {len(df)} valid {TIME_STEP}-min samples with positive AC generation")
    
    # Specify daily outputs
    
    # NREL/TP-5200-57991, 2013 Weather Corrected Performance Ratio (April 2013)
    T_cell_avg = pd.concat((df["module_temp_1__760"],
                            df["module_temp_2__761"]))
    #print('mod_temp1', df["module_temp_1__760"])
    #print('mod_temp2', df["module_temp_2__761"])
    T_cell_avg = T_cell_avg.groupby(T_cell_avg.index).mean() # average temp of modules at each index
    STC_day = P_STC*(df["poa_irradiance"]/G_STC)*(1 - TEST_THERMAL_CONST*(AMBIENT_TEMP_MEAN-T_cell_avg)) 
    result = {
        "year": year*np.ones_like(df["ac_power__752"].to_numpy()),
        "month": month*np.ones_like(df["ac_power__752"].to_numpy()),
        "EN_ac": df["ac_power__752"].to_numpy(), # hourly energy generation
        "EN_stc_corrected":STC_day.to_numpy(), # W-h weather-corrected estimate of power output 
        "T_cell_avg":T_cell_avg.to_numpy(), # degree C average cell temperature
        "poa_irradiance":df["poa_irradiance"].to_numpy(), # W/m2
    }
    
    # drop NaNs
    
    mask = ~np.isnan(np.column_stack(list(result.values()))).any(axis=1)
    result = {
        key: arr[mask]
        for key, arr in result.items()
    }

    # The raw daily dataframe is discarded when this function returns.
    # Result = dictionary of np arrays
    return result


# Main import process

def build_dataset(system_id, start_year, end_year):

    keys, irradiances = get_pvdaq_files(
        system_id,
        start_year,
        end_year
    )

    print(f"\nFound {len(keys)} files.\n")
    
    results = []
    
    for i, key in enumerate(keys, start=1): # iterate over all files from pvdaq set    
        print(f"[{i}/{len(keys)}]")

        # get daily data
        try:
            result = process_file(key, irradiances)
            if not any(np.asarray(v).size == 0 for v in result):
                results.append(result) # np array of dictionaries of np arrays ;-;

        except Exception as e:
            print(f"ERROR processing:")
            print(key)
            print(e)
            
    print("results",results)
    # Convert the collection of daily summaries
    # into a dataframe.
    hourly = []
    for d in results:
        try:
            hourly.append(pd.DataFrame(d))
        except:
            continue # skip any remaining NaNs
    
    hourly = pd.DataFrame(pd.concat(hourly)).reset_index(drop=True)
    #print(hourly.columns)
    #print(hourly.dtypes)
    #print(hourly.head())
    #print("hourly\n",hourly,"\n-------")
    # return sums over all days in each month
    monthly = hourly.groupby(["year","month"]).agg({
        "EN_ac": "sum",
        "EN_stc_corrected": "sum",
        "T_cell_avg":"mean",
        "poa_irradiance":"mean",
    })
    
    monthly["performance ratio"] = (monthly["EN_ac"] / monthly["EN_stc_corrected"])
    
    monthly = monthly.drop(columns=["EN_stc_corrected"])
    print("monthly\n", monthly,"\n========")
    #print("day_AC_gen_shape",monthly["EN_ac"].shape)
    #print("day_STC_AC_gen_shape",monthly["EN_stc_corrected"].shape)
    return monthly

# Run

result_data = build_dataset(
    system_id=SYSTEM_ID,
    start_year=START_YEAR,
    end_year=END_YEAR
)

# Sort chronologically
result_data = result_data.reset_index()
result_data = result_data.sort_values(
    ["year", "month"]
).reset_index(drop=True)


print("\nFinal dataset:")
print(result_data)

print("\nShape:")
print(result_data.shape)


# Save the reduced dataset
# CSV 
result_data.to_csv(
    f"pv_summary_{START_YEAR}_{END_YEAR}.csv",
    index=False
)

# # Parquet (may be faster)
# result_data.to_parquet(
#     "pvdaq_summary.parquet",
#     index=False
# )