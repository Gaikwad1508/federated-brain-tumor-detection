import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 📊 DATA EXTRACTED FROM YOUR LOGS (Do not change)
data = {
    'Round': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Accuracy': [
        24.89, 87.91, 91.32, 88.69, 90.97, 89.57, 
        90.97, 90.01, 91.41, 90.89, 90.27, 92.29, 
        91.32, 91.67, 91.41, 91.94
    ],
    'Loss': [
        1.530, 0.349, 0.265, 0.330, 0.294, 0.349, 
        0.345, 0.378, 0.307, 0.373, 0.339, 0.328, 
        0.397, 0.388, 0.423, 0.380
    ]
}

df = pd.DataFrame(data)

# Setup the style
sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# PLOT 1: ACCURACY
plt.subplot(1, 2, 1)
sns.lineplot(x='Round', y='Accuracy', data=df, marker='o', linewidth=3, color='#2ecc71')
plt.title('Global Model Accuracy (15 Rounds)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Communication Round', fontsize=12)
plt.ylim(0, 100)

# Highlight the Peak
peak_acc = max(df['Accuracy'])
peak_round = df[df['Accuracy'] == peak_acc]['Round'].values[0]
plt.annotate(f'Peak: {peak_acc}%', xy=(peak_round, peak_acc), xytext=(peak_round, peak_acc+5),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, fontweight='bold')

# PLOT 2: LOSS
plt.subplot(1, 2, 2)
sns.lineplot(x='Round', y='Loss', data=df, marker='o', linewidth=3, color='#e74c3c')
plt.title('Global Model Loss', fontsize=14, fontweight='bold')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Communication Round', fontsize=12)

plt.tight_layout()
plt.savefig("final_federated_results.png", dpi=300)
print(f"✅ Success! Graph saved. Peak Accuracy: {peak_acc}% at Round {peak_round}")
plt.show()