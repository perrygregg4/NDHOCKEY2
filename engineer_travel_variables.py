import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import pytz
from datetime import datetime

# Load the data
df = pd.read_csv('/Users/perrygregg/Downloads/NDHockey.csv')

# ===== 1. CALCULATE OPPONENT SCORE AND SCORING MARGIN =====
print("Calculating opponent score and scoring margin...")

# For each game, find the opponent's score
df['opponent_score'] = df.groupby('game_id')['team_score'].transform(
    lambda x: x.iloc[0] if len(x) == 2 and x.iloc[1] != x.iloc[0] else (x.iloc[1] if len(x) == 2 else np.nan)
)

# Recalculate more carefully - for each row, find the OTHER team's score in same game
def get_opponent_score(row):
    game_id = row['game_id']
    team_id = row['team_id']
    game_data = df[(df['game_id'] == game_id) & (df['team_id'] != team_id)]
    if len(game_data) > 0:
        return game_data.iloc[0]['team_score']
    return np.nan

df['opponent_score'] = df.apply(get_opponent_score, axis=1)
df['scoring_margin'] = df['team_score'] - df['opponent_score']

print(f"✓ Added opponent_score and scoring_margin")
print(f"  Sample margins: {df['scoring_margin'].head(10).tolist()}")

# ===== 2. CALCULATE DISTANCE TRAVELED (Haversine formula) =====
print("\nCalculating travel distance between consecutive games...")

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in miles
    """
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return np.nan
    
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3959  # Radius of earth in miles
    return c * r

# Sort by team and game date
df_sorted = df.sort_values(['team_id', 'game_date']).reset_index(drop=True)

# Calculate distance to previous game for each team
df_sorted['travel_distance'] = np.nan
for team in df_sorted['team_id'].unique():
    team_data = df_sorted[df_sorted['team_id'] == team].index
    for i in range(1, len(team_data)):
        curr_idx = team_data[i]
        prev_idx = team_data[i-1]
        
        curr_row = df_sorted.loc[curr_idx]
        prev_row = df_sorted.loc[prev_idx]
        
        distance = haversine_distance(
            prev_row['venue_lat'], prev_row['venue_lon'],
            curr_row['venue_lat'], curr_row['venue_lon']
        )
        df_sorted.loc[curr_idx, 'travel_distance'] = distance

# Merge back to original order
df = df_sorted.sort_index()

print(f"✓ Added travel_distance")
print(f"  Travel distances (miles): {df[df['travel_distance'].notna()]['travel_distance'].describe()}")

# ===== 3. CALCULATE REST DAYS =====
print("\nCalculating rest days between games...")

df_sorted = df.sort_values(['team_id', 'game_date']).reset_index(drop=True)

df_sorted['rest_days'] = np.nan
for team in df_sorted['team_id'].unique():
    team_data = df_sorted[df_sorted['team_id'] == team].index
    for i in range(1, len(team_data)):
        curr_idx = team_data[i]
        prev_idx = team_data[i-1]
        
        curr_date = pd.to_datetime(df_sorted.loc[curr_idx, 'game_date'])
        prev_date = pd.to_datetime(df_sorted.loc[prev_idx, 'game_date'])
        
        rest = (curr_date - prev_date).days
        df_sorted.loc[curr_idx, 'rest_days'] = rest

df = df_sorted.sort_index()

print(f"✓ Added rest_days")
print(f"  Rest days: {df[df['rest_days'].notna()]['rest_days'].describe()}")

# ===== 4. CALCULATE TIMEZONE SHIFT =====
print("\nCalculating timezone shifts...")

def get_timezone_offset_hours(tz_string):
    """Get UTC offset for timezone string"""
    try:
        tz = pytz.timezone(tz_string)
        offset = tz.localize(datetime.now()).strftime('%z')
        hours = int(offset[:3])
        return hours
    except:
        return 0

df_sorted = df.sort_values(['team_id', 'game_date']).reset_index(drop=True)
df_sorted['timezone_offset'] = df_sorted['local_time_zone'].apply(get_timezone_offset_hours)

df_sorted['timezone_shift'] = np.nan
for team in df_sorted['team_id'].unique():
    team_data = df_sorted[df_sorted['team_id'] == team].index
    for i in range(1, len(team_data)):
        curr_idx = team_data[i]
        prev_idx = team_data[i-1]
        
        curr_offset = df_sorted.loc[curr_idx, 'timezone_offset']
        prev_offset = df_sorted.loc[prev_idx, 'timezone_offset']
        
        shift = abs(curr_offset - prev_offset)
        # Handle wraparound (e.g., -8 to +8 should be 16, but really 8)
        if shift > 12:
            shift = 24 - shift
        
        df_sorted.loc[curr_idx, 'timezone_shift'] = shift

df = df_sorted.sort_index()

print(f"✓ Added timezone_shift")
print(f"  Timezone shifts (hours): {df[df['timezone_shift'].notna()]['timezone_shift'].describe()}")

# ===== 5. CREATE TRAVEL INTENSITY INDEX =====
print("\nCreating travel intensity index...")

# Normalize each component (0-1 scale)
df['travel_distance_norm'] = (df['travel_distance'] - df['travel_distance'].min()) / \
                              (df['travel_distance'].max() - df['travel_distance'].min())

df['rest_days_norm'] = 1 - ((df['rest_days'] - df['rest_days'].min()) / \
                             (df['rest_days'].max() - df['rest_days'].min()))  # Inverse: less rest = higher stress

df['timezone_shift_norm'] = df['timezone_shift'] / df['timezone_shift'].max()

# Combined index (weighted average)
# Higher values = more travel stress
df['travel_intensity'] = (
    0.4 * df['travel_distance_norm'] +  # 40% distance
    0.4 * df['rest_days_norm'] +        # 40% lack of rest
    0.2 * df['timezone_shift_norm']     # 20% timezone shift
)

print(f"✓ Added travel_intensity (composite index)")
print(f"  Travel intensity: {df[df['travel_intensity'].notna()]['travel_intensity'].describe()}")

# ===== 6. CREATE HOME/AWAY INDICATOR =====
print("\nAdding home/away indicator...")
df['is_away'] = (~df['is_home']).astype(int)

# ===== 7. CATEGORIZE TRAVEL INTENSITY =====
print("\nCategorizing travel intensity...")

def categorize_intensity(intensity):
    if pd.isna(intensity):
        return 'First Game'
    elif intensity < 0.33:
        return 'Low'
    elif intensity < 0.67:
        return 'Medium'
    else:
        return 'High'

df['travel_intensity_category'] = df['travel_intensity'].apply(categorize_intensity)

# ===== 8. SUMMARY STATISTICS =====
print("\n" + "="*70)
print("ENGINEERED VARIABLES SUMMARY")
print("="*70)

print("\nTravel Distance (miles):")
print(df[df['travel_distance'].notna()]['travel_distance'].describe())

print("\nRest Days:")
print(df[df['rest_days'].notna()]['rest_days'].describe())

print("\nTimezone Shift (hours):")
print(df[df['timezone_shift'].notna()]['timezone_shift'].describe())

print("\nTravel Intensity (0-1 scale):")
print(df[df['travel_intensity'].notna()]['travel_intensity'].describe())

print("\nTravel Intensity Categories:")
print(df['travel_intensity_category'].value_counts())

print("\nScoring Margin:")
print(df['scoring_margin'].describe())

print("\nWin Rate by Travel Intensity:")
win_by_travel = df.groupby('travel_intensity_category')['is_win'].agg(['sum', 'count', 'mean'])
win_by_travel.columns = ['Wins', 'Total Games', 'Win_Rate']
win_by_travel['Win %'] = (win_by_travel['Win_Rate'] * 100).astype(float).round(1)
print(win_by_travel[['Wins', 'Total Games', 'Win %']])

print("\nAvg Scoring Margin by Travel Intensity:")
margin_by_travel = df.groupby('travel_intensity_category')['scoring_margin'].agg(['mean', 'std', 'count'])
margin_by_travel.columns = ['Avg Margin', 'Std Dev', 'Games']
print(margin_by_travel)

# ===== 9. SAVE ENRICHED DATASET =====
print("\n" + "="*70)
output_file = '/Users/perrygregg/Downloads/NDHockey_enriched.csv'
df.to_csv(output_file, index=False)
print(f"✓ Saved enriched dataset to: {output_file}")
print(f"  Total records: {len(df)}")
print(f"  New columns: travel_distance, rest_days, timezone_shift, timezone_offset,")
print(f"               travel_intensity, travel_intensity_category, opponent_score,")
print(f"               scoring_margin, is_away")
print("="*70)
