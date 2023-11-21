import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import streamlit.components.v1 as components

def blackScholes(S, K, r, T, sigma, type="c"):
    "Calculate Black Scholes option price for a call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)

        return price
    except:  
        st.sidebar.error("Please confirm all option parameters!")


def optionDelta (S, K, r, T, sigma, type="c"):
    "Calculates option delta"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            delta = norm.cdf(d1, 0, 1)
        elif type == "p":
            delta = -norm.cdf(-d1, 0, 1)

        return delta
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionGamma (S, K, r, T, sigma):
    "Calculates option gamma"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        gamma = norm.pdf(d1, 0, 1)/ (S * sigma * np.sqrt(T))
        return gamma
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionTheta(S, K, r, T, sigma, type="c"):
    "Calculates option theta"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            theta = - ((S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2, 0, 1)

        elif type == "p":
            theta = - ((S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r*T) * norm.cdf(-d2, 0, 1)
        return theta/365
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionVega (S, K, r, T, sigma):
    "Calculates option vega"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        vega = S * np.sqrt(T) * norm.pdf(d1, 0, 1) * 0.01
        return vega
    except:
        st.sidebar.error("Please confirm all option parameters!")

def optionRho(S, K, r, T, sigma, type="c"):
    "Calculates option rho"
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    try:
        if type == "c":
            rho = 0.01 * K * T * np.exp(-r*T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            rho = 0.01 * -K * T * np.exp(-r*T) * norm.cdf(-d2, 0, 1)
        return rho
    except:
        st.sidebar.error("Please confirm all option parameters!")

def calculate_pnl(initial_price, K, r, T, sigma, type="c"):
    # Calculate option price only once for the given K, using the initial_price
    option_price_at_purchase = blackScholes(initial_price, K, r, T, sigma, type)
    
    # Calculate P&L for all spot prices
    pnl = []
    for S_current in spot_prices:
        if type == "c":
            # For a call option, the P&L is Current Asset Price - Strike Price - option_price_at_purchase
            pnl_point = S_current - K - option_price_at_purchase if S_current - K > option_price_at_purchase else -option_price_at_purchase
        elif type == "p":
            # For a put option, the P&L is Strike Price - Current Asset Price - option_price_at_purchase
            pnl_point = K - S_current - option_price_at_purchase if K - S_current > option_price_at_purchase else -option_price_at_purchase
        pnl.append(pnl_point)
    
    return pnl




st.set_page_config(page_title="Black-Scholes Calculator with charts and tables")

sidebar_title = st.sidebar.header("Black-Scholes Parameters")
space = st.sidebar.header("")
r = st.sidebar.number_input("Risk-Free Rate", min_value=0.000, max_value=1.000, step=0.001, value=0.000, format="%.3f")
S = st.sidebar.number_input("Underlying Asset Price", min_value=0.10, step=0.10, value=3000.00)
K = st.sidebar.number_input("Central Strike Price", min_value=1.00, step=0.10, value=2000.00)

# New sidebar input for Strike Price Step ($)
strike_price_step = st.sidebar.number_input("Strike Price Step ($)", min_value=1, value=50, step=1)

# Underlying Price at Option Purchase($) (Initial Price)
initial_price = st.sidebar.number_input("Underlying Price at Option Purchase($)", min_value=0.0, value=1900.00, step=0.01, format="%.2f")


days_to_expiry = st.sidebar.number_input("Time to Expiry Date (in days)", min_value=0, step=1, value=9)
hours_to_expiry = st.sidebar.number_input("Time to Expiry Date (in hours)", min_value=0, max_value=23, step=1, value=21)
minutes_to_expiry = st.sidebar.number_input("Time to Expiry Date (in minutes)", min_value=0, max_value=59, step=1, value=26)
sigma = st.sidebar.number_input("Volatility", min_value=0.0000, max_value=1.0000, step=0.0001, value=0.4989, format="%.4f")
type_input = st.sidebar.selectbox("Option Type",["Call", "Put"])

# New code block to calculate strike prices
central_strike = K  # Your central strike price from the sidebar
num_strikes = 5  # Number of strikes above and below the central strike
strike_prices = [central_strike + i * strike_price_step for i in range(-num_strikes, num_strikes + 1)]

# New code block for creating checkbox controls for P&L lines
pnl_visibility = {K: True for K in strike_prices}  # Dictionary to keep track of visibility


type=""
if type_input=="Call":
    type = "c"
elif type_input=="Put":
    type = "p"

total_days_to_expiry = days_to_expiry + hours_to_expiry / 24 + minutes_to_expiry / (24 * 60)
T = total_days_to_expiry/365

st.sidebar.text(f"Time to Expiry (years): T={T:.5f}")
st.sidebar.text(f"Volatility (in percent): {sigma * 100:.2f}%")

spot_prices = [i for i in range(0, int(S)+50 + 1)]

# First, calculate the option price at purchase for each strike price outside the loop
option_prices_at_purchase = {K: blackScholes(initial_price, K, r, T, sigma, type) for K in strike_prices}

# Calculate P&L for each strike price
# This is a dictionary comprehension that creates a dictionary where each strike price is a key and the value is the P&L list
pnl_data = {K: calculate_pnl(initial_price, K, r, T, sigma, type) for K in strike_prices}

prices = [blackScholes(i, K, r, T, sigma, type) for i in spot_prices]
# New code block: Calculate BEP and insert new Matplotlib and Plotly charts for BEP here
# Insert the BEP calculation code here
break_even_price = [K + p if type == "c" else K - p for p in prices]
deltas = [optionDelta(i, K, r, T, sigma, type) for i in spot_prices]
gammas = [optionGamma(i, K, r, T, sigma) for i in spot_prices]
thetas = [optionTheta(i, K, r, T, sigma, type) for i in spot_prices]
vegas = [optionVega(i, K, r, T, sigma) for i in spot_prices]
rhos = [optionRho(i, K, r, T, sigma, type) for i in spot_prices]

sns.set_style("whitegrid")

fig1, ax1 = plt.subplots()
sns.lineplot(x=spot_prices, y=prices)
ax1.set_ylabel('Option Price')
ax1.set_xlabel("Underlying Asset Price")
ax1.set_title("Option Price")

fig_bep, ax_bep = plt.subplots()
ax_bep.plot(spot_prices, break_even_price, color='green')
ax_bep.set_title('Break Even Price')
ax_bep.set_xlabel('Underlying Asset Price')
ax_bep.set_ylabel('Break Even Price')

fig2, ax2 = plt.subplots()
sns.lineplot(x=spot_prices, y=deltas)
ax2.set_ylabel('Delta')
ax2.set_xlabel("Underlying Asset Price")
ax2.set_title("Delta")

fig3, ax3 = plt.subplots()
sns.lineplot(x=spot_prices, y=gammas)
ax3.set_ylabel('Gamma')
ax3.set_xlabel("Underlying Asset Price")
ax3.set_title("Gamma")

fig4, ax4 = plt.subplots()
sns.lineplot(x=spot_prices, y=thetas)
ax4.set_ylabel('Theta')
ax4.set_xlabel("Underlying Asset Price")
ax4.set_title("Theta")

fig5, ax5 = plt.subplots()
sns.lineplot(x=spot_prices, y=vegas)
ax5.set_ylabel('Vega')
ax5.set_xlabel("Underlying Asset Price")
ax5.set_title("Vega")

fig6, ax6 = plt.subplots()
sns.lineplot(x=spot_prices, y=rhos)
ax6.set_ylabel('Rho')
ax6.set_xlabel("Underlying Asset Price")
ax6.set_title("Rho")

fig1.tight_layout()
fig_bep.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()


st.markdown("<h2 align='center'>Black-Scholes Option Price Calculator</h2>", unsafe_allow_html=True)
st.markdown("<h5 align='center'>Forked from Tiago Moreira's Github Repo with love <3 </h5>", unsafe_allow_html=True)
st.header("")
st.markdown("<h6>Modified for ETH price and Deribit Options (Slightly different from Deribit Option Pricer due to lack of Rho). T value is also different, don't know how they get that T value lol. Example of Deribit's Option Pricer data with Time to Expiry (01 Dec 2023): 9d 21h 26m (Weekly): <a href='https://pasteboard.co/2pFIp5OryStz.png'>https://pasteboard.co/2pFIp5OryStz.png</a></h6> So for call option price, it's $2 less expensive than calculation using this tool due to lower T value (and maybe Rho, who knows lol). Feedback are more than welcome, please create a Github Issue for feedback. See project's description and assumptions here: <a href='github.com/nonameless88/Black-Scholes-Calculator-With-Charts-Tables/'>github.com/nonameless88/Black-Scholes-Calculator-With-Charts-Tables/</a></h6>", unsafe_allow_html=True)
st.markdown("<h3 align='center'>Option Prices and Greeks</h3>", unsafe_allow_html=True)
st.header("")

# Extract the specific values for the current underlying asset price
call_price = blackScholes(S, K, r, T, sigma, type="c")
put_price = blackScholes(S, K, r, T, sigma, type="p")
delta = optionDelta(S, K, r, T, sigma, type)
gamma = optionGamma(S, K, r, T, sigma)
theta = optionTheta(S, K, r, T, sigma, type)
vega = optionVega(S, K, r, T, sigma)
rho = optionRho(S, K, r, T, sigma, type)
# Calculate break_even_price based on the type
break_even_price = [K + p if type == "c" else K - p for p in prices]

# First row of metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Call Price", f"{call_price:.2f}")
col2.metric("Put Price", f"{put_price:.2f}")
col3.metric("Break Even Price", f"{break_even_price[-1]:.2f}")

# Second row of metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Delta", f"{delta:.4f}")

# Third row of metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Gamma", f"{gamma:.10f}")

# Fourth row of metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Theta", f"{theta:.3f}")

col1, col2 = st.columns(2)
col1.metric("Vega", f"{vega:.5f}")
col2.metric("Rho", f"{rho:.5f}") 

st.header("")
st.markdown("<h3 align='center'>Visualization of the Greeks</h3>", unsafe_allow_html=True)
st.header("")
st.pyplot(fig1)
st.pyplot(fig_bep)
st.pyplot(fig2)
st.pyplot(fig3)
st.pyplot(fig4)
st.pyplot(fig5)
st.pyplot(fig6)

#---------------------------------------------------------#
# Interactive charts using Plotly
#---------------------------------------------------------#

# Create a trace for the Option Price chart
option_price_trace = go.Scatter(
    x=spot_prices,
    y=prices,
    mode='lines+markers',
    name='Option Price',
    hoverinfo='text',  # Change to 'text' to use custom text
    hovertemplate='<i>Underlying Asset Price</i>: %{x:.2f}'+
                  '<br><b>Option Price</b>: %{y:.10f}<extra></extra>',  # Custom hover text
)

# Create the layout for the chart
layout = go.Layout(
    title='Option Price Interactive Chart',
    xaxis=dict(title='Underlying Asset Price'),
    yaxis=dict(title='Option Price'),
    hovermode='closest'
)

# Create the figure with the trace and layout
fig7 = go.Figure(data=[option_price_trace], layout=layout)

# Display the figure in the Streamlit app
st.plotly_chart(fig7)

# After calculating the option prices and break-even prices
prices = [blackScholes(i, K, r, T, sigma, type) for i in spot_prices]
break_even_prices = [K + p if type == "c" else K - p for p in prices]

# This will create a list of lists, where each inner list contains the break-even price
# and the corresponding option price
hover_data = [[bep, op] for bep, op in zip(break_even_prices, prices)]


# Interactive Chart using Plotly for Break Even Price
bep_trace = go.Scatter(
    x=spot_prices,
    y=break_even_price,
    mode='lines+markers',
    name='Break Even Price',
    hoverinfo='text',  # Enabling custom hover text
    customdata=hover_data,  # Set customdata to hover_data
    hovertemplate=(
        '<i>Underlying Asset Price</i>: %{x:.2f}' +
        '<br><b>Break Even Price</b>: %{y:.2f}' +
        '<br><b>Option Price</b>: %{customdata[1]:.2f}' +
        '<br><b>Strike Price</b>: ' + str(K) + '<extra></extra>'# Custom hover text
    )
)
fig_bep_interactive = go.Figure(data=[bep_trace], layout=go.Layout(
    title='Break Even Price Interactive Chart',
    xaxis=dict(title='Underlying Asset Price'),
    yaxis=dict(title='Break Even Price'),
    hovermode='closest'
))
st.plotly_chart(fig_bep_interactive)

# Delta Chart
delta_trace = go.Scatter(
    x=spot_prices,
    y=deltas,
    mode='lines+markers',
    name='Delta',
    hoverinfo='text',
    hovertemplate='<i>Underlying Asset Price</i>: %{x:.2f}'+
                  '<br><b>Delta</b>: %{y:.10f}<extra></extra>',
)
fig8 = go.Figure(data=[delta_trace], layout=go.Layout(
    title='Delta Interactive Chart',
    xaxis=dict(title='Underlying Asset Price'),
    yaxis=dict(title='Delta'),
    hovermode='closest'
))
st.plotly_chart(fig8)

# Gamma Chart
gamma_trace = go.Scatter(
    x=spot_prices,
    y=gammas,
    mode='lines+markers',
    name='Gamma',
    hoverinfo='text',
    hovertemplate='<i>Underlying Asset Price</i>: %{x:.2f}'+
                  '<br><b>Gamma</b>: %{y:.10f}<extra></extra>',
)
fig9 = go.Figure(data=[gamma_trace], layout=go.Layout(
    title='Gamma Interactive Chart',
    xaxis=dict(title='Underlying Asset Price'),
    yaxis=dict(title='Gamma'),
    hovermode='closest'
))
st.plotly_chart(fig9)

# Theta Chart
theta_trace = go.Scatter(
    x=spot_prices,
    y=thetas,
    mode='lines+markers',
    name='Theta',
    hoverinfo='text',
    hovertemplate='<i>Underlying Asset Price</i>: %{x:.2f}'+
                  '<br><b>Theta</b>: %{y:.10f}<extra></extra>',
)
fig10 = go.Figure(data=[theta_trace], layout=go.Layout(
    title='Theta Interactive Chart',
    xaxis=dict(title='Underlying Asset Price'),
    yaxis=dict(title='Theta'),
    hovermode='closest'
))
st.plotly_chart(fig10)

# Vega Chart
vega_trace = go.Scatter(
    x=spot_prices,
    y=vegas,
    mode='lines+markers',
    name='Vega',
    hoverinfo='text',
    hovertemplate='<i>Underlying Asset Price</i>: %{x:.2f}'+
                  '<br><b>Vega</b>: %{y:.10f}<extra></extra>',
)
fig11 = go.Figure(data=[vega_trace], layout=go.Layout(
    title='Vega Interactive Chart',
    xaxis=dict(title='Underlying Asset Price'),
    yaxis=dict(title='Vega'),
    hovermode='closest'
))
st.plotly_chart(fig11)

# Rho Chart
rho_trace = go.Scatter(
    x=spot_prices,
    y=rhos,
    mode='lines+markers',
    name='Rho',
    hoverinfo='text',
    hovertemplate='<i>Underlying Asset Price</i>: %{x:.2f}'+
                  '<br><b>Rho</b>: %{y:.10f}<extra></extra>',
)
fig12 = go.Figure(data=[rho_trace], layout=go.Layout(
    title='Rho Interactive Chart',
    xaxis=dict(title='Underlying Asset Price'),
    yaxis=dict(title='Rho'),
    hovermode='closest'
))
st.plotly_chart(fig12)

# New code block for creating P&L chart
pnl_traces = []  # List to store all P&L traces
for K in strike_prices:
    if pnl_visibility[K]:  # Only add the trace if the checkbox for it is checked
        option_price_at_purchase = option_prices_at_purchase[K]  # Get the option price at purchase for this strike price
        pnl_trace = go.Scatter(
            x=spot_prices,
            y=pnl_data[K],
            mode='lines',
            name=f"P&L for Strike {K}",
            hoverinfo='text',
            text=[
                f'Strike Price: {K}<br>Underlying Price: {Si}<br>Option Price at Purchase: {option_price_at_purchase:.2f}<br>P&L: {pnl:.2f}'
                for Si, pnl in zip(spot_prices, pnl_data[K])
            ],
            hovertemplate='%{text}<extra></extra>',
        )
        pnl_traces.append(pnl_trace)

# Create the P&L figure and add all the traces
pnl_figure = go.Figure(data=pnl_traces)
pnl_figure.update_layout(
    title='P&L Interactive Chart',
    xaxis=dict(title='Underlying Asset Price'),
    yaxis=dict(title='Profit & Loss'),
    hovermode='closest'
)

# Display the P&L figure in the Streamlit app
st.plotly_chart(pnl_figure)

# Space before the new checkboxes
st.write("\n\n")

# Initialize an empty dictionary to store checkbox states
pnl_visibility = {}

# Define the number of checkboxes per row
checkboxes_per_row = 5

# Calculate the number of rows needed for all checkboxes
num_rows = len(strike_prices) // checkboxes_per_row + (len(strike_prices) % checkboxes_per_row > 0)

# Create the checkbox rows
for row in range(num_rows):
    # Create a row of columns for checkboxes
    cols = st.columns(checkboxes_per_row)
    # For each column in the current row, create a checkbox if there is a strike price available
    for i in range(checkboxes_per_row):
        # Calculate the index of the strike price
        strike_index = row * checkboxes_per_row + i
        if strike_index < len(strike_prices):
            # Create the checkbox in the respective column
            with cols[i]:
                strike = strike_prices[strike_index]
                # Create a checkbox and store the state in the pnl_visibility dictionary
                pnl_visibility[strike] = st.checkbox(f"Show P&L for Strike {strike}", True, key=f"checkbox_{strike}")


#---------------------------------------------------------#
#CREATING TABLES#
#---------------------------------------------------------#

# Create DataFrames for each set of data
table1_data = pd.DataFrame({
    'Underlying Asset Price': spot_prices,
    'Option Price': [f"{x:.10f}" for x in prices]
})
# Insert Data Table for Break Even Price Here
table_bep_data = pd.DataFrame({
    'Underlying Asset Price': spot_prices,
    'Break Even Price': break_even_price
})
table2_data = pd.DataFrame({
    'Underlying Asset Price': spot_prices,
    'Delta': [f"{x:.10f}" for x in deltas]
})
table3_data = pd.DataFrame({
    'Underlying Asset Price': spot_prices,
    'Gamma': [f"{x:.10f}" for x in gammas]
})
table4_data = pd.DataFrame({
    'Underlying Asset Price': spot_prices,
    'Theta': [f"{x:.10f}" for x in thetas]
})
table5_data = pd.DataFrame({
    'Underlying Asset Price': spot_prices,
    'Vega': [f"{x:.10f}" for x in vegas]
})
table6_data = pd.DataFrame({
    'Underlying Asset Price': spot_prices,
    'Rho': [f"{x:.10f}" for x in rhos]
})
# Custom CSS to ensure tables take up the maximum available width
st.markdown("""
<style>
.css-1l02zno {max-width:100% !important;}
</style>
""", unsafe_allow_html=True)

# Custom CSS to inject into Streamlit's HTML to adjust table styling
st.markdown("""
<style>
table {
    width: 100%;
}
th {
    text-align: left;
}
th, td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}
tr:hover {
    background-color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

# Concatenate all the tables by the 'Underlying Asset Price' column
tablefinal_data = pd.concat(
    [table1_data.set_index('Underlying Asset Price'),
     table_bep_data.set_index('Underlying Asset Price'),
     table2_data.set_index('Underlying Asset Price'),
     table3_data.set_index('Underlying Asset Price'),
     table4_data.set_index('Underlying Asset Price'),
     table5_data.set_index('Underlying Asset Price'),
     table6_data.set_index('Underlying Asset Price')],
    axis=1)

# Reset index so 'Underlying Asset Price' becomes a column again
tablefinal_data = tablefinal_data.reset_index()

# Display the final table 
st.write('Final Data Table')
st.dataframe(tablefinal_data.style.set_properties(**{'text-align': 'left'}))

# Now display the other DataFrames as tables in Streamlit with the formatted values
st.write('Option Price Data Table')
st.dataframe(table1_data.style.set_properties(**{'text-align': 'left'}))
st.write('Break Even Price Data Table')
st.dataframe(table_bep_data)
st.write('Delta Data Table')
st.dataframe(table2_data.style.set_properties(**{'text-align': 'left'}))
st.write('Gamma Data Table')
st.dataframe(table3_data.style.set_properties(**{'text-align': 'left'}))
st.write('Theta Data Table')
st.dataframe(table4_data.style.set_properties(**{'text-align': 'left'}))
st.write('Vega Data Table')
st.dataframe(table5_data.style.set_properties(**{'text-align': 'left'}))
st.write('Rho Data Table')
st.dataframe(table6_data.style.set_properties(**{'text-align': 'left'}))

# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Capture the option type and strike price inputs
option_type = type_input  # Assuming this is the variable holding the option type selected by the user
strike_price = K  # Assuming this is the variable holding the strike price entered by the user

# Export CSV buttons for each table to include the option type and strike price
# Export CSV buttons for Option Price Data Table
csv = convert_df_to_csv(table1_data)
st.download_button(
    label="Export Option Price Data Table as CSV",
    data=csv,
    file_name=f'Option_Price_Data_Table_{option_type}_{strike_price:.0f}.csv',
    mime='text/csv',
)

# Export CSV Button for Break Even Price Data Table
csv = convert_df_to_csv(table_bep_data)
st.download_button(
    label="Export Break Even Price Data Table as CSV",
    data=csv,
    file_name=f'Break_Even_Price_Data_Table_{type_input}_{K:.2f}.csv',
    mime='text/csv',
)

# Export CSV buttons for Delta Data Table
csv = convert_df_to_csv(table2_data)
st.download_button(
    label="Export Delta Data Table as CSV",
    data=csv,
    file_name=f'Delta_Data_Table_{option_type}_{strike_price:.0f}.csv',
    mime='text/csv',
)
# Export CSV buttons for Gamma Data Table
csv = convert_df_to_csv(table3_data)
st.download_button(
    label="Export Gamma Data Table as CSV",
    data=csv,
    file_name=f'Gamma_Data_Table_{option_type}_{strike_price:.0f}.csv',
    mime='text/csv',
)
# Export CSV buttons for Theta Data Table
csv = convert_df_to_csv(table4_data)
st.download_button(
    label="Export Theta Data Table as CSV",
    data=csv,
    file_name=f'Theta_Data_Table_{option_type}_{strike_price:.0f}.csv',
    mime='text/csv',
)

# Export CSV buttons for Vega Data Table
csv = convert_df_to_csv(table5_data)
st.download_button(
    label="Export Vega Data Table as CSV",
    data=csv,
    file_name=f'Vega_Data_Table_{option_type}_{strike_price:.0f}.csv',
    mime='text/csv',
)

# Export CSV buttons for Rho Data Table
csv = convert_df_to_csv(table6_data)
st.download_button(
    label="Export Rho Data Table as CSV",
    data=csv,
    file_name=f'Rho_Data_Table_{option_type}_{strike_price:.0f}.csv',
    mime='text/csv',
)

# And the final table
csv = convert_df_to_csv(tablefinal_data)
st.download_button(
    label="Export Final Data Table as CSV",
    data=csv,
    file_name=f'Final_Data_Table_{option_type}_{strike_price:.0f}.csv',
    mime='text/csv',
)

# Assuming pnl_data is a dictionary with strike prices as keys and lists of P&L values as values
table_pnl_data = pd.DataFrame({
    'Underlying Asset Price': spot_prices,
})

# Add P&L columns to the DataFrame, formatted to two decimal places
for K, pnl in pnl_data.items():
    table_pnl_data[f'P&L for Strike {K}'] = [f"{pnl_value:.2f}" for pnl_value in pnl]

# Display the table in Streamlit
st.write('P&L Data Table')
st.dataframe(table_pnl_data)

def convert_df_to_csv(df):
    # Round the P&L values to two decimals before converting to CSV
    for col in df.columns:
        if 'P&L for Strike' in col:
            df[col] = df[col].apply(lambda x: f"{float(x):.2f}")
    return df.to_csv(index=False).encode('utf-8')

# Export CSV Button for P&L Data Table
csv_pnl = convert_df_to_csv(table_pnl_data)
st.download_button(
    label="Export P&L Data Table as CSV",
    data=csv_pnl,
    file_name=f'PNL_Data_Table_{option_type}_{central_strike}_{num_strikes*2+1}_{strike_price_step}.csv',
    mime='text/csv',
)

