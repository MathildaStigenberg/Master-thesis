import pandas as pd
import sys
import matplotlib.pyplot as plt
import plotly.express as px

# optuna hyperparameters 
df = sys.argv[1]

output = sys.argv[2]

df_new = df.drop(axis=0, index=0)

columns = []
for i in df_new.columns:
    if i != 'eval_loss':
        columns.append(i)

fig = px.parallel_categories(df_new, dimensions=columns, color='eval_loss', color_continuous_scale=px.colors.sequential.Inferno, width=1350, title="Model performance for different combinations of parameters", dimensions_max_cardinality=5)
fig.write_html(output+'params.html')

fig, ax = plt.subplots()
plt.scatter(df.latent_dim, df.eval_loss, alpha=0.5)
plt.title('Model performance for different dimensions of the latent space')
plt.ylabel('Evaluation loss')
plt.xlabel('Dimension')
plt.savefig(output+'dim_latent_space.png')
