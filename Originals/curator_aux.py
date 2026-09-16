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




def df_type(df):
    if "Collection" in df.columns:
        df_type="2"
    else:
        if "location" in df.columns:
           df_type="3"
        else: 
           df_type ="1"

    return df_type

def quality_check (df: pd.DataFrame) -> bool:
    if df_type(df)=="2":
        stat_columns = ["Collection","Latitude","Longitude","N","Ns","Cutoff","S","Dec","Inc",
                        "R","k","a95","K","A95","A95Min","A95Max","ΔDx","ΔIx","λ","Pole Lng","Pole Lat"]
        if not all(col in df.columns for col in stat_columns):
                print("Missing values. Please check your data, or export statistics again.")		
        if df.duplicated("Collection").any():
                print("Data contains duplicate rows. Please remove duplicates.")
        for idx, row in df.iterrows():
            if all(pd.notna(row[col]) for col in stat_columns) and row["k"] > 5:
                return True  
            else:
                return False
            
    if df_type(df)=="1":
        comp_columns = ['Dec', 'Inc', 'a95', 'k', 'Lat', 'Long']
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
            if row["nSample"] > 4 or row["nSites"] > 1 and row["k"] > 5:
                return True
            else:
                print("Data fails basic quality control. Please check your data.")
                return False
            
    if df_type(df)=="3":
        mag_columns = ["site",
                        "lat",
                        "lon",
                        "dir_k",
                        "dir_alpha95",
                        "dir_dec",
                        "dir_inc",
                        "age_unit"]
        if not all(col in df.columns for col in mag_columns):
                print("Missing values. Please check your data, or export statistics again.")
        

def import_data(path):
    df=pd.read_csv(path)
    if quality_check(df):
        print("Data imported successfully, and passes quality control. Congratulations!")
    else:
        print("Data fails quality control. Please check your data.")
    return df

def import_magic_data(path):
    data_list, file_type = pmag.magic_read(path)
    df = pd.DataFrame(data_list)
    

def translator(df):
    # if df_type(df)=="1":
        df["Notes"] = ""
        for idx, row in df.iterrows():
            if pd.isna(row["nSample"]):
                df.at[idx, "nSample"] = row["Nsites"]
                df.at[idx, "Notes"] = "nSample was NaN, set to Nsites value."
        keep_columns = ["AgeUpBound",
                        "AgeLowBound",
                        "Primary",
                        "MaxAgeMag",
                       "MinAgeMag",
                        "nSample",
                        "Dec",
                        "Inc",
                        "a95",
                        "k",
                        "Lat",
                        "Long",
                        "Ref",
                        "A95_VGP",
                        "K_VGP",
                        "Notes",
                        "age"
                        ]
        translated_df=df[keep_columns].copy()
        translated_df = translated_df.rename(columns={
            "ID": "name",
            "nSample":"N",
            "Lat": "slat",
            "Long": "slon",
            "Dec":"mdec",
            "Inc":"minc",
            "A95_VGP":"A95",
            "K_VGP":"K"
            })
        translated_df['Primary'] = translated_df['Primary'].map({'Y': True, 'N': False})
        translated_df["AgeUpBound"] = pd.to_numeric(translated_df["AgeUpBound"], errors="coerce")
        translated_df["AgeLowBound"] = pd.to_numeric(translated_df["AgeLowBound"], errors="coerce")

        missing_age_mask = translated_df["age"].isna() | (translated_df["age"] == "")
        print(missing_age_mask.sum(), "rows with missing age information found. Attempting to fill in missing ages...")

        for idx, row in translated_df.loc[missing_age_mask].iterrows():
            if row["Primary"]:
                translated_df.loc[idx, "age"] = (row["AgeUpBound"] + row["AgeLowBound"]) / 2
                translated_df.rename(columns={"AgeUpBound": "max_age", "AgeLowBound": "min_age"}, inplace=True)
                print("Filled missing age for primary data at index", idx)
            else:
                translated_df.loc[idx, "age"] = (row["MaxAgeMag"] + row["MinAgeMag"]) / 2
                translated_df.loc[idx, "max_age"] = row["MaxAgeMag"]
                translated_df.loc[idx, "min_age"] = row["MinAgeMag"]
                print("Filled missing age for non-primary data at index", idx)
        for idx, row in translated_df.loc[missing_age_mask].iterrows():
            if row["min_age"] == row["age"]:
                translated_df.loc[idx, "min_age"] = row["min_age"]-0.01
            if row["max_age"] == row["age"]:
                translated_df.loc[idx, "max_age"] = row["max_age"]+0.01       
   #    translated_df = translated_df.drop(columns=["MaxAgeMag", "MinAgeMag","Primary"])
        translated_df = translated_df.drop(columns=["Primary"])
        return translated_df

    # if df_type(df)=="2":
    #     translation_map = {"Collection": "name",
    #                     "Lat": "slat",
    #                     "Long": "slon",
    #                     "K_VGP":"K",
    #                     "A95_VGP":"A95",
    #                     "Dec":"mdec",
    #                     "Inc":"minc",
    #                     "Pole Lng":"plon",
    #                     "Pole Lat":"plat",
    #                     "nSample":"N"}
    #     translated_df = df.rename(columns=translation_map)
    #     return translated_df        

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

def curate_ages(df):
    df_age=df.copy()

    if df_type(df) == "2":
        age_data_path = input("Please provide the path to the CSV file with age data: ")
        age_data = pd.read_csv(age_data_path)
        df_age = pd.merge(df_age, age_data, on="name", how="left")
        print("Age data added. Remaining rows:", df_age.shape[0])

    if df_age["age"].isnull().any():
        remove_missing = input("Warning: Some rows are missing age information. Please check your data. " \
        "Do you want to remove rows with missing age information? (Y/N): ").strip().lower()
        if remove_missing == 'y' or "Y":
            df_age = df_age.dropna(subset=["age"])
            print("Rows with missing age information removed.")
        else:
            print("Rows with missing age not removed. Please check your data.")

    #ADD PERIOD data to the df
    #I think we should make it more granular -- maybe by EPOCH or even AGE???

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
    for idx, row in df_age.iterrows():
        
        age = pd.to_numeric(row["age"], errors='coerce')
        found = False
        for period, (start, end) in geological_periods.items():
            if start <= age < end:
                df_age.loc[idx, "period"] = period
                found = True
                break
        if not found:
            df_age.loc[idx, "period"] = "Unknown"
    
    if "period" in df_age.columns and (df_age["period"] == "Unknown").any():
      i = input("Some directions have Unknown periods. Do you want to remove this data? (Y/N) ")
      if i.strip().lower() == "y":
        df_age = df_age[df_age["period"] != "Unknown"].copy()
        df_age.reset_index(drop=True, inplace=True)
        i=input("Some directions have Unknown periods. Do you want to remove this data? (Y/N)")
        if i =="Y" or "y":
            for idx, row in df_age.iterrows():
                if df_age["period"]=="Unknown":
                    df_age.drop(idx, inplace=True)
            df_age.reset_index(drop=True, inplace=True)

    return df_age

def cleaning(df):
    df_clean=df.copy()

    df = df.dropna(subset=['N'])

    print(df_clean.columns)

    #Step 1: remove rows without a95 and k values; or k values > 100
    df_clean=df_clean.dropna(subset=["a95","k"])
    #df_clean = df_clean[df_clean["k"] <= 100]

    #Step 2: calculate A95 and K values based on Deenen et al., 2011 formula
    df_clean["lambda"] = ""
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
            
    print("Final cleaning complete. Remaining rows:", df_clean.shape[0])
    return df_clean

def pole_and_rotation_export(df, p1, p2):
    df.to_csv(p1, index=False)
    df.to_csv(p2, index=False)

def curator(df):
    tec_data=df.copy()
    tec_data=translator(df)
    tec_data=convert_polarity(tec_data)
    tec_data=curate_ages(tec_data)
    tec_data=cleaning(tec_data)
    print("Data curated successfully. Please use the Calculator script to obtain rotation data before plotting.")
    return tec_data






    
