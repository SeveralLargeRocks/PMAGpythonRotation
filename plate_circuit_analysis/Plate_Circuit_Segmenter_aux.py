import pandas as pd
import math
import os
import numpy as np

def calculate_rot_matrix_from_euler_pole(lat, lon, angle):
    # Converts an euler pole into a rotation matrix using ROdriguez formaulation
    lat = math.radians(lat)
    lon = math.radians(lon)
    theta = math.radians(angle)

    # Unit axis vector
    kx = math.cos(lat) * math.cos(lon)
    ky = math.cos(lat) * math.sin(lon)
    kz = math.sin(lat)

    c = math.cos(theta)
    s = math.sin(theta)
    v = 1 - c

    R = [
        [c + v*kx*kx,     v*kx*ky - s*kz, v*kx*kz + s*ky],
        [v*ky*kx + s*kz,  c + v*ky*ky,    v*ky*kz - s*kx],
        [v*kz*kx - s*ky,  v*kz*ky + s*kx, c + v*kz*kz]
    ]

    return R

def search_for_plateIDs(target_id, anchor_ID, plate_circuit):

    plate_circuit = plate_circuit.copy()
    plate_circuit['plate_id'] = plate_circuit['plate_id'].astype(str).str.zfill(3)
    plate_circuit['rel_plate'] = plate_circuit['rel_plate'].astype(str).str.zfill(3)

    target_id = str(target_id)
    anchor_ID = str(anchor_ID)

    IDPath = [target_id]
    loop_id = target_id

    while loop_id != anchor_ID:

        match = plate_circuit[plate_circuit['plate_id'] == loop_id]

        if match.empty:
            raise ValueError(f"Plate ID {loop_id} not found in circuit")

        parent_id = match.iloc[0]['rel_plate']

        IDPath.append(parent_id)
        loop_id = parent_id

    return IDPath
      
# def collect_EP_from_IDPath(IDPath, plate_circuit, target_age):
#     #Collects euler pole rotations along a specified IDPath up to a target age
#     #Collects full euler poles for any age below target age
#     #collects partial interpolated poles for age ranges containing target_age
#     EP_list = []
#     for 

def split_rotation_files(rotation_file, output_folder):
    #takes a rotation file and segments it into 1ma increments
    for index, row in rotation_file.iterrows():
        start_age = row['age']
        end_age = row['age'].shift(+1)
        total_rotation = row['angle']

        increment = total_rotation / (end_age - start_age) 

def interpolate_rotation_files(directory):
    for filename in os.listdir(directory):
        raw_rotation_file = pd.read_csv(os.path.join(directory, filename))
        raw_rotation_file['age'] = raw_rotation_file['age'].round().astype(int)
        raw_rotation_file = raw_rotation_file.drop_duplicates(subset='age', keep='first')
        max_age = raw_rotation_file['age'].max()
        ages = np.arange(0, max_age + 1, 1)

        df = pd.DataFrame({
            'plate_id': pd.Series(dtype='int'),
            'age': ages,
            'lat': pd.Series(dtype='float'),
            'lon': pd.Series(dtype='float'),
            'angle': pd.Series(dtype='float'),
            'rel_plate': pd.Series(dtype='int')})
        
        for row in df.itertuples():
            age = row.age
            stage = raw_rotation_file[
                raw_rotation_file['age'] >= age
            ].sort_values('age').iloc[0]

            df.at[row.Index, 'plate_id'] = stage['plate_id']
            df.at[row.Index, 'lat'] = stage['lat']
            df.at[row.Index, 'lon'] = stage['lon']
            df.at[row.Index, 'angle'] = stage['angle']
            df.at[row.Index, 'rel_plate'] = stage['rel_plate']