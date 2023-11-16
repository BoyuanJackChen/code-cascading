import pandas as pd

import plotly.express as px

threshold = 0.1
df = pd.read_csv(f'full_threshold{threshold}.csv')

def is_pareto(cost, accuracy, costs, accuracies):
    for c, a in zip(costs, accuracies):
        if c <= cost and a >= accuracy and (c < cost or a > accuracy):
            return False
    return True
df['Pareto'] = df.apply(lambda x: is_pareto(x['cost'], x['accuracy'], df['cost'], df['accuracy']), axis=1)

fig = px.scatter(df, x='cost', y='accuracy', color='Pareto',
                 hover_data=['k1', 'k2', 'k3', 't1', 't2', 't3'],
                 color_discrete_map={True: 'green', False: 'lightblue'})

fig.update_layout(title='fWizardCoder-Python-V1.0 Family on HumanEval, pick@0,1,3,5,10, testlines=2,4, threshold={threshold}',
                  xaxis_title='Cost',
                  yaxis_title='Accuracy')

fig.show()
