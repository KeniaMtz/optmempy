import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_csv("technoeconomic_temperature_results.csv")

world = gpd.read_file("world-countries.json")
world = world.rename(columns={'id': 'iso_alpha'})
merged = world.merge(df, on="iso_alpha", how="left")

#maptype = "co2"
#maptype = "energy"
maptype = "cost"

if maptype == "co2":
  column = "co2_red_kton_y"
  label = "CO₂ reduction (1000 ton/year)"
  title = "National CO₂-Equivalent Emission Reduction Potential"
  filename = "co2_world_map"

  vmin = 0
  vmax = 70

elif maptype == "energy":
  column = "energy_red_gwh_y"
  label = "Energy reduction (GWh/year)"
  title = "National Energy Reduction Potential"
  filename = "energy_world_map"

  vmin = 0
  vmax = 70

elif maptype == "cost":
  column = "cost_red_musd_y"
  label = "Operating cost reduction (million USD/year)"
  title = "National Operating Cost Reduction Potential"
  filename = "cost_world_map"

  vmin = 0
  vmax = 10

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

world.plot(
  ax=ax,
  color="#afafaf",
  edgecolor="#000000",
  linewidth=0.5
)

merged.dropna(subset=[column]).plot(
  column=column,
  ax=ax,
  cmap='plasma',
  vmin=vmin,
  vmax=vmax,
  edgecolor='white',
  linewidth=0.5
)

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap='plasma')
sm._A = []

cax = fig.add_axes([0.30, 0.2, 0.40, 0.025])
cax.set_facecolor('white')

cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

cbar.set_label(label, fontsize=16)

cbar.ax.tick_params(labelsize=14)

ax.axis('off')

plt.savefig(filename + ".png", dpi=300, bbox_inches='tight')
plt.show()