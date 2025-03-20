import pandas as pd

df = pd.read_excel('Compiled Result.xlsx')


# win rate > 40
# final balance > 4000
# max drawdown < 30

df = df[(df['Win Rate'] > 30) & (df['Final_Balance'] > 2000) & (df['Max Drawdown'] < 30)]

print(len(df))


import plotly.express as px

fig = px.scatter_3d(df, x='fast', y='medium', z='slow', title='3D Scatter Plot')

fig.update_layout(scene=dict(
    xaxis_title='Fast',
    yaxis_title='Medium',
    zaxis_title='Slow'
))

fig.show()
