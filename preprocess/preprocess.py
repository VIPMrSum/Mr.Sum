import os
import json
import math
import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm
from yt8m_reader import read_tfrecord

def align_most_replayed(file_name, random_id, youtube_id, duration):
    mostreplayed_json_path = f"mostReplayed/{file_name}/{file_name}_{random_id}_{youtube_id}.json"
    
    aligned = []
    with open(mostreplayed_json_path, 'r') as fd:
        data = json.load(fd)
        mr_chunk_size = data['heatMarkers'][0]["heatMarkerRenderer"]["markerDurationMillis"]
        for n in range(int(duration)):
            bin_number = math.floor(n * 1000 / mr_chunk_size)
            if bin_number > 99.01:
                bin_number = 99
            
            aligned.append(data['heatMarkers'][bin_number]["heatMarkerRenderer"]["heatMarkerIntensityScoreNormalized"])
    
    return np.array(aligned)

def preprocess():
    meta_data = "dataset/meta_data.csv"
    h5fd = h5py.File("dataset/mrsum.h5", 'a')
    df = pd.read_csv(meta_data)
    
    for row in tqdm(df.itertuples()):
        feature, labels = read_tfrecord(row.yt8m_file, row.random_id)
        h5fd.create_dataset(f"{row.video_id}/features", data=feature)
        
        # mostreplayed = align_most_replayed(row.yt8m_file, row.random_id, row.youtube_id, row.duration)
        # h5fd.create_dataset(f"{row.video_id}/gtscore", data=mostreplayed)
    
    h5fd.close()

if __name__ == "__main__":
    preprocess()