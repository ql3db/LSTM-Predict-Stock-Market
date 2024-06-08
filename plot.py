import matplotlib.pyplot as plt
import pandas as pd
import math
import yfinance as yf

#############
# Figure 1
#############

# Download S&P 500 data
sp500_data = yf.download('^GSPC', start='2020-12-31', end='2024-05-20')

# Save the data to a CSV file
sp500_data.to_csv('sp500_data.csv')

# Load historical S&P 500 data (replace 'sp500_data.csv' with your data file)
sp500_data = pd.read_csv('sp500_data.csv')

# Assuming your data has columns 'Date' and 'Close'
dates = pd.to_datetime(sp500_data['Date'])
prices = sp500_data['Close']

# Highlight CPI release dates for 2023
cpi_dates = [
    '2023-02-10', '2023-03-10', '2023-04-11', '2023-05-10', '2023-06-09', 
    '2023-07-11', '2023-08-10', '2023-09-12', '2023-10-11', '2023-11-09', 
    '2023-12-08', '2024-01-10', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-10'
]

fed_dates = ['2022-03-16','2023-12-12']

# Plot S&P 500 prices
plt.figure(figsize=(12, 6))
plt.plot(dates, prices, label='S&P 500')

# Highlight CPI release dates for 2023
plt.axvline(x=pd.to_datetime(fed_dates[0]), color='g', linestyle='-.', label='Fed start hiking')
plt.axvline(x=pd.to_datetime(fed_dates[1]), color='r', linestyle='-.', label='Fed pivot')

plt.xlabel('Date',fontsize=18)
plt.ylabel('Price',fontsize=18)
plt.title('Figure 1: S&P 500 Prices Since 2021',fontsize=20)
plt.legend(fontsize=16)
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.subplots_adjust(bottom=0.25)
plt.savefig("snp500.png")
plt.tight_layout()
plt.show()

#############
# Figure 3 &4
#############

# Load the data from the CSV file
csv_file = 'loss_dict.csv'
cap_csv_file = 'cap_loss_dict.csv'
df = pd.read_csv(csv_file)
cap_df = pd.read_csv(cap_csv_file)

# Compute the average losses for each industry (assuming each row represents a loss value)
average_losses = df.mean().to_dict()
cap_average_losses = cap_df.mean().to_dict()

# Function to plot the average losses
def plot_industry_loss(losses):
    # Sort the dictionary by values
    sorted_losses = sorted(losses.items(), key=lambda x: x[1])

    # Unpack sorted items
    industries, values = zip(*sorted_losses)

    # Process labels to insert newline character between words
    processed_labels = ['\n'.join(industry.split()) for industry in industries]

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(processed_labels)), values, color='#89CFF0')
    ax.set_title('Figure 3: Average LSTM Losses by Industry',fontsize=20)
    ax.set_xlabel('Industry',fontsize=18)
    ax.set_ylabel('Average Losses',fontsize=18)

    # Set custom x-axis labels with rotation
    ax.set_xticks(range(len(processed_labels)))
    ax.set_xticklabels(processed_labels, rotation=90,fontsize=18)  # Adding rotation here
    plt.yticks(fontsize=18)

    # Enable horizontal grid lines only
    ax.yaxis.grid(True)  # Enable horizontal grid lines
    ax.xaxis.grid(False)  # Disable vertical grid lines

    # Optional: Adjust subplot parameters or use tight_layout to fit labels
    plt.subplots_adjust(bottom=0.25)  # Increase bottom margin to better accommodate rotated labels
    plt.tight_layout()

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.4f}', ha='center', va='bottom',fontsize=16)

    # Save and show the plot
    plt.savefig('average_losses_chart.png', dpi=300)
    plt.show()

def plot_cap_loss(losses):
    # Sort the dictionary by values
    sorted_losses = sorted(losses.items(), key=lambda x: x[1])

    # Unpack sorted items
    caps, values = zip(*sorted_losses)

    # Process labels to insert newline character between words
    processed_labels = ['\n'.join(cap.split()) for cap in caps]

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(processed_labels)), values, color='#89CFF0')
    ax.set_title('Figure 4: Average LSTM Losses by Cap Category', fontsize=20)
    ax.set_xlabel('Cap Category', fontsize=18)
    ax.set_ylabel('Average Losses', fontsize=18)

    # Set custom x-axis labels with rotation
    ax.set_xticks(range(len(processed_labels)))
    ax.set_xticklabels(processed_labels, rotation=90, fontsize=18)
    plt.yticks(fontsize=18)

    # Enable horizontal grid lines only
    ax.yaxis.grid(True)  # Enable horizontal grid lines
    ax.xaxis.grid(False)  # Disable vertical grid lines

    # Optional: Adjust subplot parameters or use tight_layout to fit labels
    plt.subplots_adjust(bottom=0.25)  # Increase bottom margin to better accommodate rotated labels
    plt.tight_layout()

    # Add value labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.4f}', ha='center', va='bottom', fontsize=16)

    # Save and show the plot
    plt.savefig('cap_average_losses_chart.png', dpi=300)
    plt.show()


# Plot the cap losses
plot_industry_loss(average_losses)
plot_cap_loss(cap_average_losses)
# print("finished")

# Find the column name of the smallest number in the row
# stock_loss_dict_df = pd.read_csv("stock_loss_dict.csv")
# min_col = stock_loss_dict_df.idxmin(axis=1)
# min_col_name = min_col.iloc[0]
# print('stock with smallest loss',min_col_name,stock_loss_dict_df[min_col_name])

# def plot_returns(returns_df, stock):
#     plt.figure(figsize=(10, 5))
#     col_actual_str = stock+'_actual'
#     col_pred_str = stock+'_pred'
#     plt.plot(returns_df.index, returns_df[col_actual_str], label=stock +' Actual Values', marker='o')
#     plt.plot(returns_df.index, returns_df[col_pred_str], label=stock +' Predicted Values', marker='x')
#     plt.title(stock + ' Actual vs Predicted Values')
#     plt.xlabel('Date')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# returns_df = pd.read_csv("returns_df.csv")
# plot_returns(returns_df,min_col_name)


# print(returns_df[['Date','AAPL_actual','AAPL_pred']])

