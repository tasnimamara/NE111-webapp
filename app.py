import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="NE 111 distribution fitting tool")
st.title("NE 111 distribution fitting tool")

choice = st.radio(
    "how would you like to enter data?",
    ["1. type numbers", "2. load from CSV file"],
    format_func=lambda x: x
)

data = None

if choice == "1. type numbers":
    st.write("\nenter numbers separated by commas or spaces:")
    user_input = st.text_area("data:", height=100)
    
    if user_input:
        numbers = []
        for x in user_input.replace(',', ' ').split():
            try:
                numbers.append(float(x.strip()))
            except:
                pass
        data = np.array(numbers)
        st.write(f"loaded {len(data)} values")
    
elif choice == "2. load from CSV file":
    uploaded_file = st.file_uploader("enter CSV filename:", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            data = df.iloc[:, 0].dropna().values
            st.write(f"loaded {len(data)} values from {uploaded_file.name}")
        except:
            st.write("error loading file!")

if data is not None and len(data) > 0:
    st.write("\n data statistics:")
    st.write(f"sample size: {len(data)}")
    st.write(f"mean: {np.mean(data):.3f}")
    st.write(f"std dev: {np.std(data):.3f}")
    st.write(f"min: {np.min(data):.3f}")
    st.write(f"max: {np.max(data):.3f}")

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(data, 'k.', alpha=0.6)
    ax1.set_xlabel('measurement #')
    ax1.set_ylabel('value')
    ax1.set_title('data Points')
    ax1.grid(True, alpha=0.3)

    ax2.hist(data, bins=20, density=True, alpha=0.7, color='lightpink', edgecolor='purple',)
    ax2.set_xlabel('value')
    ax2.set_ylabel('frequency')
    ax2.set_title('data histogram')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig1)

    st.write("\n available distributions:")
    distributions = [
        'normal', 'gamma', 'exponential', 'log-normal', 'weibull',
        'beta', 'uniform', 'chi-squared', 'rayleigh', 'cauchy'
    ]

    for i, dist in enumerate(distributions, 1):
        st.write(f"   {i}. {dist}")

    choice_idx = st.selectbox(
        f"choose distribution number (1-{len(distributions)}):",
        range(1, len(distributions)+1),
        format_func=lambda x: f"{x}. {distributions[x-1]}"
    ) - 1
    
    try:
        selected_dist = distributions[choice_idx]
    except:
        st.write("not a valid choice, using choice 1: normal")
        selected_dist = 'normal'

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
    st.write(f"\n fitting {selected_dist} distribution")
    params = dist_class.fit(data)
    fitted_dist = dist_class(*params)

    st.write(f" finished {selected_dist} distribution!")
    st.write("\n fitted parameters:")
    for i, param in enumerate(params, 1):
        st.write(f"   parameter {i}: {param:.6f}")

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
    st.pyplot(fig2)

    st.write("\n  quality of fit metrics:")
    hist, bin_edges = np.histogram(data, bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    predicted = fitted_dist.pdf(bin_centers)

    mse = np.mean((hist - predicted) ** 2)
    max_error = np.max(np.abs(hist - predicted))

    st.write(f"   mean squared error: {mse:.6f}")
    st.write(f"   maximum error: {max_error:.6f}")

    st.write("\n manual fitting option")
    st.write("adjust parameters manually:")

    manual_params = []
    cols = st.columns(len(params))
    for i, (col, param) in enumerate(zip(cols, params), 1):
        with col:
            new_param = st.number_input(
                f"parameter {i} (current {param:.6f}):",
                value=float(param),
                format="%.6f"
            )
            manual_params.append(new_param)

    manual_dist = dist_class(*manual_params)
    manual_pdf = manual_dist.pdf(x)

    fig3, ax4 = plt.subplots(figsize=(8, 6))

    ax4.hist(data, bins=20, density=True, alpha=0.7, 
            color='lightpink', edgecolor='purple', label='data histogram')
    ax4.plot(x, pdf, 'r-', linewidth=2, label=f'best fit {selected_dist}')
    ax4.plot(x, manual_pdf, 'g--', linewidth=2, label=f'manual {selected_dist}')

    ax4.set_xlabel('value')
    ax4.set_ylabel('density')
    ax4.set_title(f'best fit vs manual fit: {selected_dist}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)

    st.write("\nmanual fit parameters:")
    for i, param in enumerate(manual_params, 1):
        st.write(f"   parameter {i}: {param:.6f}")

    hist, bin_edges = np.histogram(data, bins=20, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    manual_predicted = manual_dist.pdf(bin_centers)

    manual_mse = np.mean((hist - manual_predicted) ** 2)
    manual_max_error = np.max(np.abs(hist - manual_predicted))

    st.write("\nmanual fit quality metrics:")
    st.write(f"   mean squared error: {manual_mse:.6f}")
    st.write(f"   maximum error: {manual_max_error:.6f}")

    st.write(" analysis complete!")

    save_choice = st.checkbox("save results to file?")
    if save_choice:
        filename = st.text_input("enter what you want the file name to be:", "results.txt")
        if filename:
            results_text = f"""distribution fitting results
{'='*40}
data points: {len(data)}
distribution: {selected_dist}

best fit parameters:
"""
            for i, param in enumerate(params, 1):
                results_text += f"parameter {i}: {param:.6f}\n"
            results_text += f"""
best fit quality metrics:
  MSE: {mse:.6f}
  max error: {max_error:.6f}

manual fit parameters:
"""
            for i, param in enumerate(manual_params, 1):
                results_text += f"parameter {i}: {param:.6f}\n"
            results_text += f"""
manual fit quality metrics:
  MSE: {manual_mse:.6f}
  max error: {manual_max_error:.6f}
"""
            st.download_button(
                label="Download results",
                data=results_text,
                file_name=filename,
                mime="text/plain"
            )
            st.write(f"results saved to {filename}")
