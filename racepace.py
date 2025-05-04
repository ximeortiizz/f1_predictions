import fastf1
import pandas as pd
from fastf1 import plotting

fastf1.Cache.enable_cache("f1_cache")
session= fastf1.get_session(2025, "Saudi Arabia", "R")
session.load()

#Obtain data lap times
laps= session.laps
laps= laps[laps['LapTime'].notnull()]

#Filter to get only one lap
lap_1_data=laps[laps['LapNumber']==1]


race_laps = laps[
    (laps['LapNumber'] > 1) &  # exclude first lap
    (laps['PitInTime'].isnull()) &  # not ending in pit
    (laps['PitOutTime'].isnull()) &  # not starting from pit
    (laps['IsAccurate'] == True)
]

if race_laps.empty:
    print("Race laps are empty")
    race_laps=laps[
        (laps['LapNumber']>1) #exclude outlaps
        & (laps['IsAccurate']==True)
    ]

print (f"Found {len(race_laps)}valid race laps.")

avg_race_pace=race_laps.groupby('Driver')['LapTime'].mean().sort_values()

#Normalize lap times by track lengths for comparison (Jedah:6.174km)
track_length_km=6.174
avg_race_pace_normalized= avg_race_pace.dt.total_seconds()/track_length_km

#create data frame with the average race pace and normalized race pace
race_pace_df=pd.DataFrame({
    'Average Race Pace':avg_race_pace.dt.total_seconds(),
    ' Average Race Pace normalized': avg_race_pace_normalized
})

print(race_pace_df)