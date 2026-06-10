import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from shapely.geometry import box

world = gpd.read_file("world-countries.json")
df_ocean = pd.read_csv("ocean_temp_data.csv")

df_ocean['lon'] = df_ocean['lon'].apply(lambda x: x - 360 if x > 180 else x)
lons_unique = sorted(df_ocean['lon'].unique())
lats_unique = sorted(df_ocean['lat'].unique())
lon_grid, lat_grid = np.meshgrid(lons_unique, lats_unique)
temp_grid = np.zeros(lon_grid.shape) - 99

for i, lat in enumerate(lats_unique):
  for j, lon in enumerate(lons_unique):
    val = df_ocean.loc[(df_ocean['lat'] == lat) & (df_ocean['lon'] == lon), 't_an']
    if not val.empty:
      temp_grid[i, j] = val.values[0]

temp_grid[temp_grid == -99] = np.nan

column = "t_an"
label = "Temperature (°C)"
title = "Global Sea Surface Temperature (WOA23)"
filename = "ocean_temp_map"
vmin, vmax = -2, 32

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

im = ax.contourf(
  lon_grid, lat_grid, temp_grid, 
  levels=50,
  cmap='RdYlBu_r', 
  vmin=vmin, vmax=vmax,
  zorder=1
)

world.plot(
  ax=ax, 
  color="#afafaf", 
  edgecolor="#000000", 
  zorder=2
)

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap='RdYlBu_r')
sm._A = []

cax = fig.add_axes([0.30, 0.16, 0.40, 0.025])
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label(label, fontsize=16)
cbar.ax.tick_params(labelsize=14)

ax.axis('off')
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])

plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
plt.show()