import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np
import csv

# Read the CSV file and create a DataFrame
df = pd.DataFrame(columns=["short", "long", "final", "drawdown", "winrate"])
try:
    with open('eth_overfit_long.csv', 'r') as FILE:
        reader = csv.reader(FILE, delimiter=';')
        next(reader)  # Skip the header
        for line in reader:
            if line:
                values = line[0].split(',')
                short, long, final, drawdown, winrate = map(float, values[:5])
                df = pd.concat([df, pd.DataFrame({'short': [short], 'long': [long], 'final': [final], 'drawdown': [drawdown], 'winrate': [winrate]})], ignore_index=True)
except FileNotFoundError:
    print("Error: The file 'btc_overfit_long.csv' was not found.")
    exit(1)
print(df)

def plot_3d_surface(ax, x, y, z, xlabel, ylabel, zlabel, title):
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
plot_3d_surface(ax1, df['short'], df['long'], df['final'], 'short', 'long', 'final', 'Final Balance')

ax2 = fig.add_subplot(132, projection='3d')
plot_3d_surface(ax2, df['short'], df['long'], df['drawdown'], 'short', 'long', 'drawdown', 'Drawdown')

ax3 = fig.add_subplot(133, projection='3d')
plot_3d_surface(ax3, df['short'], df['long'], df['winrate'], 'short', 'long', 'winrate', 'Win Rate')

plt.tight_layout()
plt.show()