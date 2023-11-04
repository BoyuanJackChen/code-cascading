import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import mplcursors

# Read CSV data
filename = 'wizard_he_1test.csv'
data = pd.read_csv(filename)

# Fill NaN values in 'Model' and 'Num GPUs Occupied' columns using forward fill method
data['Model'] = data['Model'].ffill()
data['Num GPUs Occupied'] = data['Num GPUs Occupied'].ffill()

# Convert k to integer and fill NaN with -1
data['k'].fillna(-1, inplace=True)
data['k'] = data['k'].astype(int)

coverage_accuracy = {
    '13B-7B': 0.253,
    '34B-7B': 0.384,
    '34B-13B': 0.372
}

# Get the greedy accuracies
greedy_accuracies = {}
conditions = [
    {"Model": "7B", "k": 0},
    {"Model": "13B", "k": 0},
    {"Model": "34B", "k": 0},
]
for condition in conditions:
    model = condition["Model"]
    k = condition["k"]
    filtered_data = data[(data['Model'] == model) & (data['k'] == k)]
    if not filtered_data.empty:
        greedy_accuracies[model] = filtered_data.iloc[0]['Final accu (%)']
    else:
        greedy_accuracies[model] = None
greedy_accu_1 = greedy_accuracies['7B']/100
greedy_accu_2 = greedy_accuracies['13B']/100
greedy_accu_3 = greedy_accuracies['34B']/100

# Generate all combinations
ks = [-1,0,1,2,3,4,5,10]
combinations = product(ks, repeat=3)
results = []

# Loop over combinations and calculate
for k1, k2, k3 in combinations:
    # If smaller model is 0, all the larger models must be -1, because our generation stops there
    if (k1==-1 and k2==-1 and k3==-1) or (k1==0 and k2>=0) or (k1==0 and k3>=0) or (k2==0 and k3>=0):
        continue
    
    def extract_values(model, k):
        if k < 0:
            return 0, 0, 0, 0, 0, 0
        def parse_value(value):
            try:
                return float(value) / 100
            except ValueError:  # Return 0 if the value is not a number (i.e. '/')
                return 0
        row = data[(data['Model'] == model) & (data['k'] <= k)].iloc[-1]
        acc = round(parse_value(row['Final accu (%)']),3)
        tpr = parse_value(row['True positive (%)']) if k>0 else acc
        tnr = parse_value(row['True negative (%)'])
        fnr = parse_value(row['False negative (%)'])
        fpr = parse_value(row['False positive (%)'])
        cost = float(row['Cost on full run ($)']) if row['Cost on full run ($)'] else 0
        return tpr, tnr, fnr, fpr, acc, cost

    tp7, tn7, fn7, fp7, accu7, cost7 = extract_values('7B', k1)
    tp13, tn13, fn13, fp13, accu13, cost13 = extract_values('13B', k2)
    tp34, tn34, fn34, fp34, accu34, cost34 = extract_values('34B', k3)
    print(f"tn7: {tn7}")
    print(f"tn13: {tn13}")
    print(f"tn34: {tn34}")
    print(f"fn7: {fn7}")
    print(f"fn13: {fn13}")
    print(f"fn34: {fn34}")
    print(f"accu7: {accu7}")
    print(f"accu13: {accu13}")
    print(f"accu34: {accu34}")
    
    # Calculate final dollar consumption ($)
    final_dollar_consumption = 0
    if k1 >= 0:
        if k2 >= 0:
            final_dollar_consumption += (tn7 + fn7) * cost13
            if k3 >= 0:
                final_dollar_consumption += (tn7 + fn7) * (tn13 + fn13) * cost34
        elif k3 >= 0:
            final_dollar_consumption += (tn7 + fn7) * cost34
    elif k2 >= 0:
        final_dollar_consumption += cost13
        if k3 >= 0:
            final_dollar_consumption += (tn13 + fn13) * cost34
    else:
        final_dollar_consumption += cost34

    # Calculate final accuracy (%)
    final_accuracy = 0
    if k1 >= 0:
        final_accuracy += accu7
        if k2 >= 0 and k3==-1:
            final_accuracy += (tn7+fn7) * coverage_accuracy['13B-7B'] * accu13/greedy_accu_2
        elif k2 >= 0 and k3>=0:
            final_accuracy += (tn7+fn7) * coverage_accuracy['13B-7B'] * accu13/greedy_accu_2 * tp13/(tp13+fn13) + (tn7 + fn7) * (tn13 + fn13) * coverage_accuracy['34B-13B'] * accu34/greedy_accu_3
        elif k3 >= 0:
            final_accuracy += (tn7+fn7) * coverage_accuracy['34B-7B'] * accu34/greedy_accu_3
    elif k2 >= 0:
        final_accuracy += accu13
        if k3 >= 0:
            final_accuracy += (tn13+fn13) * coverage_accuracy['34B-13B'] * accu34/greedy_accu_3
    else:
        final_accuracy += accu34
        
    results.append((final_dollar_consumption, final_accuracy, k1, k2, k3))

    # if final_accuracy < 0.5:
    #     print(f"(k1={k1}, k2={k2}, k3={k3}); accu7={accu7}, accu13={accu13}, accu34={accu34}; {final_dollar_consumption}, {final_accuracy}")
    #     input()

# Sort results by final dollar consumption for better visualization
results.sort(key=lambda x: x[0])

# Extract data for plotting
x = [i[0] for i in results]  # Final Dollar Consumption
y = [i[1] for i in results]  # Final Accuracy

# Plotting
fig, ax = plt.subplots(figsize=(10,6))
sc = ax.scatter(x, y, picker=True)  # Use scatter plot to enable mplcursors
ax.set_xlabel('Total Dollar Consumption ($) on finishing 164 questions in HumanEval (1 set)', size=15)
ax.set_ylabel('Final Accuracy (%)', size=15)
ax.set_title('Dollar-Accuracy for all k1,k2,k3 combinations of WizardCoder Family on HumanEval', size=17)
ax.grid(True)

# Adding mplcursors for interactivity
cursor = mplcursors.cursor(sc, hover=True)

@cursor.connect("add")
def on_add(sel):
    i = sel.target.index  # Get the index of the selected point
    dollars, accuracy, k1, k2, k3 = results[i]  # Extract k1, k2, k3 from the results
    sel.annotation.set_text(f'k1={k1}, k2={k2}, k3={k3}, ${round(dollars,7)}, {round(accuracy*100,2)}%')  # Set the annotation text

plt.show()