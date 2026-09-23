import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import folium
import math
import time
import sys
import geopy
import numpy as np
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster, PolyLineTextPath
from shapely.geometry import MultiPolygon
from shapely.ops import split, unary_union
from statistics import mean,median
from pmagpy import pmag
from scipy.optimize import brentq


def quality_check_magic (df: pd.DataFrame) -> bool:
        comp_columns = ['site',
                        'lat',
                        'lon', 
                        'age', 
                        'dir_dec',
                        'dir_inc',
                        'dir_alpha95',	
                        'dir_n_samples']
        
        if not all(col in df.columns for col in comp_columns):
            print("Data is missing required columns. Please check your data.")
        if df.duplicated().any():
            print("Data contains duplicate rows. Please remove duplicates.")
        for idx, row in df.iterrows():
            if all(pd.notna(row[col]) for col in comp_columns):
                t=input("Data contains missing values in required columns. Please check your data. Do you wish to remove missing values? Y/N")
                if t=="Y" or "y":
                    df=df.dropna(subset=comp_columns)
                else:
                    return False

#A95 to k conversion based on Butler (2004, Magnetic Domains to Geologic Terranes)
def a95_from_k(k, N):
    R = (N - 1) / k
    return np.degrees(
        np.arccos(1 - ((N - R) / R) * ((1 / 0.05)**(1/(N-1)) - 1))
    )
    
def k_from_a95(a95, N):
    R = pmag.calculate_r(a95, N)
    k = pmag.calculate_k(R, N)
    return k

def add_age_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" not in df.columns:
        df["age"] = np.nan  # or np.nan if you prefer numeric
        print("Added 'age' column.")
    else:
        print("'age' column already exists.")
    return df

def add_pole_k_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pole_k" not in df.columns:
        df["pole_k"] = np.nan  # or np.nan if you prefer numeric
        print("Added 'pole_k' column.")
    else:
        print("'pole_k' column already exists.")
    return df

def add_N_alt_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "N_alt" not in df.columns:
        df["N_alt"] = np.nan  # or np.nan if you prefer numeric
        print("Added 'N_alt' column.")
    else:
        print("'N_alt' column already exists.")
    return df

def import_data(path):
    df=pd.read_csv(path)
    df = add_age_column(df)
    return df

def translator(df, custom_map_needed, custom_map):
    df = df.copy()
    df["Notes"] = ""
    add_age_column (df)
    add_pole_k_column(df)

    if custom_map_needed:
        translation_map = custom_map
    else:
        translation_map = {
            "dir_n_samples": "N",
            "lat_n": "slat",
            "lon_w": "slon",
            "dir_k": "k",
            "dir_alpha95": "a95",
            "dir_dec": "mdec",
            "dir_inc": "minc",
            "location": "loc",
            "site_alternatives": "nSites",
            "age_low": "AgeUpBound",
            "age_high": "AgeLowBound",
            "site": "name",
            "citations": "Ref",
            "pole_alpha95": "A95",
            "pole_k": "K",
            "dir_n_sites": "N_alt"
        }

    translated_df = df.rename(columns=translation_map)
    return translated_df  

def convert_polarity(df: pd.DataFrame) -> pd.DataFrame:
    norm_df=df.copy()
    norm_df["Polarity"] = 0
    norm_df['Polarity'] = norm_df.apply(lambda row: 'Normal' if row['minc'] >= 0 else 'Reverse', axis=1)
    print ("Nr. of sites with normal polarity:", (norm_df["Polarity"] == "Normal").sum())
    print ("Nr. of sites with reverse polarity:", (norm_df["Polarity"] == "Reverse").sum())

    #Ask if the user wants to convert reverse polarity to normal polarity
    if (norm_df["Polarity"] == "Reverse").sum() != 0:
        convert = input("We found some directions with reverse polarity. Do you want to convert to normal polarity? (Y/N)").strip().lower()
        if convert == 'Y' or "y":
            converted_count = 0
            for idx, row in norm_df.iterrows():
                if row["Polarity"] == "Normal":
                    continue
                elif row["Polarity"] == "Reverse":
                    norm_df.at[idx, "minc"] = -row["minc"]
                    norm_df.at[idx, "mdec"] = row["mdec"] + 180
                    norm_df.at[idx, "mdec"] = row["mdec"] % 360
                    norm_df.at[idx, 'Polarity'] = 'Normal'
                    converted_count += 1 
            print("Number of rows converted to normal polarity:", converted_count)
        else:
            print("Skipping conversion of reverse polarity.")
    return norm_df

def drop_overrepresented_locations(df, location_removal_threshold, lat_col="slat", lon_col="slon",):
    """
    Drops ALL rows whose (lat, lon) occurs more than `max_allowed` times.
    Example: if max_allowed=1, any duplicate site is removed entirely.
    """

    # Count occurrences of each lat/lon pair
    counts = df.groupby([lat_col, lon_col]).size().reset_index(name="count")

    print("Number of unique locations before removal:", counts.shape[0])

    # Identify the lat/lon pairs that occur too many times
    bad_locs = counts[counts["count"] > location_removal_threshold][[lat_col, lon_col]]

    # Merge to flag rows to drop
    df = df.merge(bad_locs.assign(drop=True), on=[lat_col, lon_col], how="left")

    # Keep only rows without the "drop" flag
    df_clean = df[df["drop"] != True].drop(columns="drop")

    return df_clean

def curate_ages(df):
    df_age=df.copy()
    
    df_age["age"] = pd.to_numeric(df_age["age"], errors="coerce")
    df_age["agehigh"] = pd.to_numeric(df_age.get("AgeUpBound"), errors="coerce")
    df_age["agelow"] = pd.to_numeric(df_age.get("AgeLowBound"), errors="coerce")

    mask_missing_age = df_age["age"].isna()
    for idx in df_age[mask_missing_age].index:
        low = df_age.loc[idx, "agelow"]
        high = df_age.loc[idx, "agehigh"]

        if pd.notna(low) and pd.notna(high):
            df_age.loc[idx, "age"] = (low + high) / 2
        else:
            df_age.loc[idx, "drop_row"] = True  # mark for removal


    # Drop rows that could not be filled
    if "drop_row" in df_age.columns:
        df_age = df_age[df_age["drop_row"] != True].drop(columns=["drop_row"])

    # Remove any remaining NaNs in age
    df_age = df_age.dropna(subset=["age"]).reset_index(drop=True)

    geological_periods = {
        "Quaternary": (0, 2.58),
        "Neogene": (2.58, 23.03),
        "Paleogene": (23.03, 66.0),
        "Cretaceous": (66.0, 145.0),
        "Jurassic": (145.0, 201.3),
        "Triassic": (201.3, 251.902),
        "Permian": (251.902, 298.9),
        "Carboniferous": (298.9, 358.9),
        "Devonian": (358.9, 419.2),
        "Silurian": (419.2, 443.8),
        "Ordovician": (443.8, 485.4),
        "Cambrian": (485.4, 541.0),
        "Precambrian": (541.0, 4600.0)}
    df_age["period"] = "Unknown"
    for idx, row in df_age.iterrows():
        age = row["age"]
        for period, (start, end) in geological_periods.items():
            if start <= age < end:
                df_age.loc[idx, "period"] = period
                break

    # Optionally remove unknown periods
    if (df_age["period"] == "Unknown").any():
       i = input("Some directions have Unknown periods. Do you want to remove this data? (Y/N) ").strip().lower()
    
       if i == "y":
            df_age = df_age[df_age["period"] != "Unknown"].reset_index(drop=True)
    return df_age

def cleaning(df,
            num_min_samples,
            location_removal_threshold,
            estimate_missing_a95_k,
            fill_sites):
    df_clean=df.copy()

    print("Dropping rows with missing a95 and k", df_clean.shape)

    df_clean = df_clean.dropna(subset=["a95", "k"], how="all")

    if estimate_missing_a95_k:
        print("Estimating missing A95 and K values", df_clean.shape)
        mask1 = (
                df_clean['a95'].isna() &
                df_clean['k'].notna() &
               (df_clean['N'] >= 5)
               )

        df_clean.loc[mask1, 'a95'] = df_clean.loc[mask1].apply(
            lambda r: a95_from_k(r['k'], r['N']),
            axis=1
        )
        mask2 = (
                df_clean['a95'].notna() &
                df_clean['k'].isna() &
               (df_clean['N'] >= 5)
               )

        df_clean.loc[mask2, 'k'] = df_clean.loc[mask2].apply(
            lambda r: k_from_a95(r['a95'], r['N']),
            axis=1
        )
        print(df_clean.shape[0], "rows remain after estimating a95 or k values.")

    else:    
        df_clean=df_clean.dropna(subset=["a95"])
        df_clean=df_clean.dropna(subset=["k"])
        print(df_clean.shape[0], "rows remain after removing rows with missing a95 or k values.")

    if fill_sites:
        df_clean = add_N_alt_column(df_clean)
        df_clean['N'] = df_clean['N'].fillna(df_clean['N_alt'])
    print("Dropping num_min_samples <", num_min_samples, df_clean.shape)
    df_clean = df_clean[df_clean['N'] >= num_min_samples]
    print(df_clean.shape[0], "rows remain after dropping rows with N <", num_min_samples)
    print("Dropping overrepresented locations with threshold >", location_removal_threshold)
    df_clean = drop_overrepresented_locations(df_clean, location_removal_threshold)

    print("Averaging locations", df_clean.shape)

    df_clean = df_clean[pd.notna(df_clean['slat']) & pd.notna(df_clean['slon'])].copy()

    #Step 2: calculate A95 and K values based on Deenen et al., 2011 formula
    df_clean["lambda"] = ""
    df_clean["lambda"] = pd.to_numeric(df_clean["lambda"], errors="coerce")


    if "K" not in df_clean.columns:
       df_clean["K"] = np.nan
    else:
       df_clean["K"] = pd.to_numeric(df_clean["K"], errors="coerce")

    if "A95" not in df_clean.columns:
        df_clean["A95"] = np.nan
    else:
        df_clean["A95"] = pd.to_numeric(df_clean["A95"], errors="coerce")

    for idx, row in df_clean.iterrows():
        df_clean.at[idx, "lambda"] = math.degrees(math.atan((math.tan(math.radians(row["minc"])))/2))

    for idx, row in df_clean.iterrows():
        if pd.isna(row["K"]):
            df_clean.at[idx, "K"] = (8*row["k"])/(5+(18*(math.sin(math.radians(row["lambda"]))**2)+ 9*(math.sin(math.radians(row["lambda"])**4))))
            df_clean.at[idx, "Notes"] += "K not given; calculated using Deenen et al., 2011 formula. "
    
    for idx, row in df_clean.iterrows():
        if pd.isna(row["A95"]):
            df_clean.at[idx, "A95"] = (140/(math.sqrt(row["N"]*row["K"])))
            df_clean.at[idx, "Notes"] += "A95 not given; calculated using Deenen et al., 2011 formula."

    print("Dropping rows without a95 or k values")
    df_clean = df_clean.dropna(subset=["a95", "k"])
            
    print("Final cleaning complete. Remaining rows:", df_clean.shape[0])
    return df_clean




def pole_and_rotation_export(df, p1, p2):
    df.to_csv(p1, index=False)
    df.to_csv(p2, index=False)

def curator(df,
            num_min_samples,
            location_removal_threshold,
            estimate_missing_a95_k,
            fill_sites,
            custom_map_needed,
            custom_map):
    tec_data=df.copy()

    print("Before translator:", tec_data.shape)

    tec_data=translator(df, custom_map_needed, custom_map)

    print("Before polarity:", tec_data.shape)

    tec_data=convert_polarity(tec_data)

    print("Before age curation:", tec_data.shape)

    tec_data=curate_ages(tec_data)

    print("Before cleaning:", tec_data.shape)

    tec_data=cleaning(tec_data,
            num_min_samples,
            location_removal_threshold,
            estimate_missing_a95_k,
            fill_sites)

    print("Data curated successfully. Please use the Calculator script to obtain rotation data before plotting.")
    return tec_data






    
