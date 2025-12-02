import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

print(" NE 111 distribution fitting tool")

print("\nhow would you like to enter data?")
print("1. type numbers ")
print("2. load from CSV file")
choice = input("enter choice (1 or 2): ")

data = None

if choice == '1':
    print("\nenter numbers separated by commas or spaces:")
    user_input = input("data: ")
    
    numbers = []
    for x in user_input.replace(',', ' ').split():
        try:
            numbers.append(float(x.strip()))
        except:
            pass
    
    data = np.array(numbers)
    print(f"loaded {len(data)} values")
    
elif choice == '2':
    filename = input("enter CSV filename: ")
    try:
        df = pd.read_csv(filename)
        data = df.iloc[:, 0].dropna().values
        print(f"loaded {len(data)} values from {filename}")
    except:
        print("error loading file!")

if data is None or len(data) == 0:
    print("no data loaded. :(")
    exit()

# Show basic statistics
print("\n Data Statistics:")
print(f"   Sample size: {len(data)}")
print(f"   Mean: {np.mean(data):.3f}")
print(f"   Std Dev: {np.std(data):.3f}")
print(f"   Min: {np.min(data):.3f}")
print(f"   Max: {np.max(data):.3f}")

# Plot 1: Show raw data (like in your assignment example)
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left plot: Raw data points
ax1.plot(data, 'k.', alpha=0.6)
ax1.set_xlabel('Measurement Number')
ax1.set_ylabel('Value')
ax1.set_title('Raw Data Points')
ax1.grid(True, alpha=0.3)

# Right plot: Histogram
ax2.hist(data, bins=15, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Data Histogram')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Ask which distribution to fit
print("\n🎯 Available Distributions:")
distributions = [
    'Normal', 'Gamma', 'Exponential', 'Log-Normal', 'Weibull',
    'Beta', 'Uniform', 'Chi-squared', 'Rayleigh', 'Cauchy'
]

for i, dist in enumerate(distributions, 1):
    print(f"   {i}. {dist}")

choice = input(f"\nChoose distribution (1-{len(distributions)}): ")
try:
    choice_idx = int(choice) - 1
    selected_dist = distributions[choice_idx]
except:
    print("❌ Invalid choice, using Normal distribution")
    selected_dist = 'Normal'

dist_map = {
    'normal': stats.norm,
    'gamma': stats.gamma,
    'exponential': stats.expon,
    'log-normal': stats.lognorm,
    'weibull': stats.weibull_min,
    'beta': stats.beta,
    'uniform': stats.uniform,
    'chi-squared': stats.chi2,
    'rayleigh': stats.rayleigh,
    'cauchy': stats.cauchy
}

dist_class = dist_map[selected_dist]
print(f"\n fitting {selected_dist} distribution")
params = dist_class.fit(data)
fitted_dist = dist_class(*params)

print(f" finished {selected_dist} distribution!")
print("\n fitted parameters:")
for i, param in enumerate(params, 1):
    print(f"   parameter {i}: {param:.6f}")

fig2, ax3 = plt.subplots(figsize=(8, 6))

ax3.hist(data, bins=20, density=True, alpha=0.7, 
        color='lightpink', edgecolor='purple', label='data histogram')

x = np.linspace(np.min(data), np.max(data), 1000)
pdf = fitted_dist.pdf(x)
ax3.plot(x, pdf, 'r-', linewidth=2, label=f'Fitted {selected_dist}')

ax3.set_xlabel('value')
ax3.set_ylabel('density')
ax3.set_title(f'data with fitted {selected_dist} distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n  quality of fit metrics:")
hist, bin_edges = np.histogram(data, bins=20, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
predicted = fitted_dist.pdf(bin_centers)

mse = np.mean((hist - predicted) ** 2)
max_error = np.max(np.abs(hist - predicted))

print(f"   mean squared error: {mse:.6f}")
print(f"   maximum error: {max_error:.6f}")

print(" analysis complete!")

save_choice = input("\nsave results to file? (y/n): ")
if save_choice.lower() == 'y':
    filename = input("enter filename (e.g., results.txt): ")
    with open(filename, 'w') as f:
        f.write(f"distribution fitting results\n")
        f.write(f"{'='*40}\n")
        f.write(f"data points: {len(data)}\n")
        f.write(f"distribution: {selected_dist}\n\n")
        f.write("fitted parameters:\n")
        for i, param in enumerate(params, 1):
            f.write(f"  Parameter {i}: {param:.6f}\n")
        f.write(f"\nquality metrics:\n")
        f.write(f"  MSE: {mse:.6f}\n")
        f.write(f"  max error: {max_error:.6f}\n")
    print(f"results saved to {filename}")