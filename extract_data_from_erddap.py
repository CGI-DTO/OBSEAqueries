#!/usr/bin/env python3
"""
Extracts data from OBSEA and generates a huge CSV file with all the data

author: Enoc Martínez
institution: Universitat Politècnica de Catalunya (UPC)
email: enoc.martinez@upc.edu
license: MIT
created: 19/09/22

Modified by Gerard Llorach
"""

from argparse import ArgumentParser
import pandas as pd
import requests
import rich
import os
import json
import numpy as np


__dataset_urls = {
    #"SBE37": "https://data.obsea.es/erddap/tabledap/OBSEA_SBE37_CTD_30min",
    #"SBE16": "https://data.obsea.es/erddap/tabledap/OBSEA_SBE16_CTD_30min",
    #"awac-waves": "https://data.obsea.es/erddap/tabledap/OBSEA_AWAC_waves",
    #"awac-currents": "https://data.obsea.es/erddap/tabledap/OBSEA_AWAC_currents_30min",
    #"Airmar_150WX": "https://data.obsea.es/erddap/tabledap/OBSEA_Buoy_Airmar_150WX_Meteo_30min"
    "CTD": "https://data.obsea.es/erddap/tabledap/OBSEA_CTD_30min",
    "awac-waves": "https://data.obsea.es/erddap/tabledap/OBSEA_AWAC_waves_full",
    "awac-currents": "https://data.obsea.es/erddap/tabledap/OBSEA_AWAC_currents_30min",
    "Airmar_150WX": "https://data.obsea.es/erddap/tabledap/OBSEA_Buoy_Airmar_150WX_30min"
}


def merge_dataframes(dataframes: list):
    """
    Concatenate dataframes into a single dataframe with the same columns
    :param dataframes:
    :return:
    """
    print(dataframes[0])
    print(dataframes[1])

    if dataframes[0].empty and dataframes[1].empty:  # if both are empty
        return dataframes[0]  # return one of them
    if dataframes[0].empty and not dataframes[1].empty:  # if df0 is empty but df1 not
        return dataframes[1]
    elif dataframes[1].empty and not dataframes[0].empty:  # if df1 is empty but df0 not
        return dataframes[0]

    df = pd.concat(dataframes)
    df = df.sort_index()
    rich.print("[yellow]Merged dataframes:")
    print(df)
    return df


def df_from_erddap(name, url, variables, start_time, end_time):
    """
    Get a dataframe from ERDDAP
    :param name:
    :param url:
    :param variables:
    :param start_time:
    :param end_time:
    :return:
    """
    __temp_csv = f"{name}_temp.csv"
    u = url + ".csv?"
    u += ",".join(variables)
    u += f"&time>={start_time}&time<={end_time}"
    resp = requests.get(u)
    if resp.status_code >= 200 and resp.status_code < 300:  # OK
        with open(__temp_csv, "w") as f:
            f.write(resp.text)

        with open(__temp_csv) as f:
            header = f.readline().strip().split(",")
            f.readline()  # skip units
            df = pd.read_csv(f, skiprows=0, names=header)
        os.remove(__temp_csv)
        df = df.rename(columns={"time": "timestamp"})

        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%SZ", utc=True)
        df = df.set_index("timestamp")
        return df
    else:
        rich.print(f"[yellow]NO DATA FOR {name} between {start_time} and {end_time}")
        return pd.DataFrame(columns=variables)


def get_erddap_metadata(url):
    metadata_url = url.replace("tabledap", "info") + "/index.json"
    print("Get", metadata_url)
    r = requests.get(metadata_url)
    metadata = json.loads(r.text)
    variables = []
    for row in metadata["table"]["rows"]:
        if row[0] == "variable":
            variables.append(row[1])
    return variables


def upsample_1h_to_30min(df):
    if df.empty:
        return df
    waves = df.resample("1h").max()
    waves2 = waves
    waves2["timestamp"] = waves2.index + pd.to_timedelta("30min")
    waves2 = waves2.set_index("timestamp")
    waves = pd.concat([waves, waves2])
    del waves["timestamp"]
    waves = waves.sort_index()
    waves = waves.dropna(how="all")
    return waves


__custom_operations = [
    #{"source": ["SBE37", "SBE16"], "operation": merge_dataframes, "destination": "CTD"},
    {"source": "awac-waves", "operation": upsample_1h_to_30min, "destination": "awac-waves"}
]


def generate_csv(start_time, end_time, file):
    """
    Generates a CSV file with ALL the data from a list of datasets
    :param start_time:
    :param end_time:
    :param file:
    :return:
    """
    # Generate a dataframe for each URL in __dataset_urls
    datasets = {}
    for name, url in __dataset_urls.items():
        print(f"Getting metadata for {name}...")
        variables = get_erddap_metadata(url)
        print(f"Generating dataset for {name} between {start_time} and {end_time}")
        df = df_from_erddap(name, url, variables, start_time, end_time)
        datasets[name] = df

    # Apply custom operations
    for operation in __custom_operations:
        source = operation["source"]
        func = operation["operation"]
        destination = operation["destination"]
        rich.print(f"operation {func.__name__} on {source}")

        if type(source) == str:
            argument = datasets[source]
            del datasets[source]
        elif type(source) == list:
            argument = [datasets[v] for v in source]  # list wit all the dataframes
            for v in source:
                del datasets[v]

        result = func(argument)  # call handler
        datasets[destination] = result  # store result

    # Create an empty dataframe with the correct indexes
    index = pd.date_range(start=start_time, end=end_time, freq="30min", tz="utc")
    index = index.set_names("timestamp")
    df_final = pd.DataFrame(index=index)

    for name, df in list(datasets.items()):
        rich.print(f"processing {name}")
        if df.empty:  # empty dataframes will be processed later
            for c in df.columns:
                df_final[c] = np.nan
        else:
            # Handle depth-based variables
            if "depth" in df.columns and df.groupby(df.index)["depth"].nunique().max() > 1:
                unique_depths = df["depth"].unique()  # Get unique depth values
                for depth in unique_depths:
                    depth_df = df[df["depth"] == depth]  # Filter rows for this depth

                    # Group by index (timestamps) and calculate the mean
                    def safe_mean(group):
                        non_nan_values = group.dropna()  # Exclude NaN values
                        unique_values = non_nan_values.nunique()  # Count unique non-NaN values
                        if unique_values > 1:
                            rich.print(
                                f"[yellow]Warning: Multiple unique values found for depth {depth} at timestamp {group.name}. "
                                f"Values: {non_nan_values.tolist()}"
                                # There is an error, for some reason there are no unique values. Check if it works here:
                                # https://data.obsea.es/erddap/tabledap/OBSEA_AWAC_currents_30min.htmlTable?time%2Clatitude%2Clongitude%2Cdepth%2CCSPD%2CCDIR%2CUCUR%2CVCUR%2CZCUR%2CCSPD_QC%2CCDIR_QC%2CUCUR_QC%2CVCUR_QC%2CZCUR_QC%2Clatitude_QC%2Clongitude_QC&time%3E=2018-04-04T19%3A00%3A00Z&time%3C=2018-04-04T19%3A00%3A00Z&depth=0&depth=0
                            )
                        return group.mean()  # Calculate the mean

                    depth_df = depth_df.groupby(depth_df.index).agg(safe_mean)

                    for var in ["UCUR", "VCUR", "ZCUR"]:  # Variables to process
                        if var in df.columns:
                            new_col_name = f"{var}_{int(depth)}m"  # Encode depth in column name
                            # Reindex to align with df_final and fill missing timestamps with NaN
                            aligned_data = depth_df[var].reindex(df_final.index, fill_value=np.nan)
                            df_final[new_col_name] = aligned_data

            
            
            
            
            
            
            # Drop unnecessary columns
            columns_to_drop = ["latitude", "longitude", "depth", "latitude_QC", "longitude_QC", "depth_QC"]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")
            
            
            try: # TODO: error for CTD for latest 6 months of 2020
                df_final = df_final.merge(df, on="timestamp", how="outer")
            except:
                rich.print("****** Error, could not merge data. ******")

    # ---- Generate custom CSV file from a list of variables ---- #
    # Variables from awac waves
    awac_waves = ["VHM0", "VAVH", "VH110", "VZMX", "VTM02", "VTPK", "VTZA", "VPED", "VMDR"]

    # Variables from CTD
    ctd = ["TEMP", "PSAL"]
    
    # Variables from Airmar_150WX
    airmar = ['CAPH', 'AIRT', 'WDIR', 'WSPD']

    # Variables from Airmar (e.g., meteorological data)
    awac_currents = ["CSPD","UCUR", "VCUR", "ZCUR"]

    # Merge all variables into a single list
    initial_cols = awac_waves + ctd + airmar + awac_currents
    columns = []
    
    # Create qc
    for c in initial_cols:  # add QC for every column in the list
        columns.append(c)
        columns.append(c + "_QC")

    # Create variables for depth values
    # for i in range(0, 20):  # Add currents columns for every depth from 0 to 19 meters
    #     columns.append(f"UCUR_{i}m")
    #     columns.append(f"UCUR_{i}m_QC")
    #     columns.append(f"VCUR_{i}m")
    #     columns.append(f"VCUR_{i}m_QC")
    #     columns.append(f"ZCUR_{i}m")
    #     columns.append(f"ZCUR_{i}m_QC")

    missing_columns = [col for col in columns if col not in df_final.columns]

    if missing_columns:
        print(f"The following columns are missing in df_final: {missing_columns}")

    df = df_final[columns]
    df = erase_data_with_bad_qc(df)
    rich.print(f"[green]Storing csv file to \"{file}\"")
    df.to_csv(file)
    rich.print("[green]done!")


def erase_data_with_bad_qc(df):
    df = df.copy()
    for var in df.columns:
        if var.endswith("_QC"):
            continue
        var_qc = var + "_QC"

        for index, row in df.loc[df[var_qc] == 4].iterrows():
            df.at[index, var] = np.nan
            df.at[index, var_qc] = np.nan

        del df[var_qc]  # erase qc variable
    return df


if __name__ == "__main__":

    # Generate 6 month data files
    # TODO: work on compression of these files
    for year in range (2018, 2019):
        # First six months
        sTime = str(year) + "-01-01T00:00:00Z"
        eTime = str(year) + "-07-01T00:00:00Z"
        fileName = "obsea_" + str(year) + "_1.csv"
        generate_csv(sTime, eTime, fileName)
        # Latest six months
        sTime = eTime
        eTime = str(year + 1) + "-01-01T00:00:00Z"
        fileName = "obsea_" + str(year) + "_2.csv"
        generate_csv(sTime, eTime, fileName)