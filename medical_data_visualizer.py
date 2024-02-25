

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Workaround to fix what seems to be problems with the version of seaborn using np.bool and np.float which are depretated
np.float = float
np.bool = bool
# Import data
df = pd.read_csv("medical_examination.csv")


# Add 'overweight' column

df["BMI"] = df["weight"]/ (df["height"]/100)**2


df['overweight'] = np.where(df["BMI"] >25, 1, 0)
print(df['overweight'].head())

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
print(df["gluc"].value_counts())
df["gluc"] = np.where(df["gluc"] == 1, 0, 1)
df["cholesterol"] = np.where(df["cholesterol"] == 1, 0, 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars = ["cardio"], value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat["variable"] = df_cat["variable"].astype('category')
    print(df_cat.head())


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    #df_cat = None


    # Draw the catplot with 'sns.catplot()'


    print(df_cat.columns)
    # Get the figure for the output
    g = sns.catplot(
      data=df_cat, x="variable", col="cardio",
      kind="count", hue="value"
  )
    g.set_axis_labels("variable", "total")
    fig = g.fig
    


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    print(fig)
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi'])
    & (df['height'] >= df['height'].quantile(0.025))
    & (df['height']<= df["height"].quantile(0.975))
    & (df["weight"]>= df["weight"].quantile(0.025))
    & (df['weight']<= df["weight"].quantile(0.975))]
    df_heat.drop(columns=['BMI'], inplace=True)
    # Calculate the correlation matrix
    
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True



    # Set up the matplotlib figure
    fig, ax = plt.subplots()
    
    

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, ax = ax, mask = mask, annot=True, center = 0, linewidths=0.5, fmt=".1f", vmax = 0.3)


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
