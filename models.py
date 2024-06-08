# pip install torch torchvision torchaudio
# pip3 install numpy
# pip3 install pandas
# pip3 install matplotlib
# pip3 install sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Function to load and preprocess data
def load_data(stock_file, industry_file, market_cap_file,macro_file):
    # Load stock price data
    stock_data = pd.read_csv(stock_file)
    industry_data = pd.read_csv(industry_file)
    market_cap_data = pd.read_csv(market_cap_file)
    macro_data = pd.read_csv(macro_file)
    macro_data.rename(columns={'Unnamed: 0':'Date'},inplace=True)
    macro_data = macro_data[['Date','CPI','Unemployment Rate']]

    #change Date
    macro_data['Date'] = pd.to_datetime(macro_data['Date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    #cpi data need % change
    macro_data.sort_values(by='Date',inplace=True)
    macro_data.dropna(inplace=True)
    # macro_data['Previous_CPI'] = macro_data['CPI'].shift(1)
    # macro_data['CPI Ratio'] = (macro_data['CPI'] / macro_data['Previous_CPI']-1)*100
    macro_data['CPI Ratio'] = macro_data['CPI'].pct_change() * 100

    #stock data need % return
    #rates need absolute change
    stock_data.sort_values(by='Date',inplace=True)
    # stock_data.dropna(inplace=True)
    stock_data.set_index('Date', inplace=True)
    stock_data.iloc[:, :-1] = stock_data.iloc[:, :-1].pct_change() * 100
    stock_data.iloc[:, -1] = stock_data.iloc[:, -1].diff()
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])    

    #only need data after 2021
    macro_data = macro_data[macro_data['Date'] >= '2020-12-31']

    #remove new issued stocks
    stock_data = stock_data.drop(columns = ['ABNB', 'CARR', 'CEG', 'GEHC', 'GEV', 'KVUE', 'OTIS', 'SOLV', 'VLTO'])

    # Combine datasets based on date and interpolate missing data
    data_df = pd.merge(stock_data, macro_data, on='Date',how='outer')
    data_df.sort_values(by='Date',inplace=True)
    data_df.fillna(method='ffill', inplace=True)  
    data_df.dropna(inplace=True)
    data_df[['Date','CPI','Unemployment Rate']].to_csv("test.csv")

    #remove weekends
    data_df = data_df[data_df['Date'].dt.dayofweek < 5]
    return data_df,industry_data, market_cap_data

def scale_data(data_df):
    cols_to_normalize = data_df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    # data_df = scaler.fit_transform(data_df)
    data_df[cols_to_normalize] = scaler.fit_transform(data_df[cols_to_normalize])
    return data_df,scaler

def create_sequences(data, sequence_length):
    X, y, dates = [], [], []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:(i + sequence_length),:]  # Capture the sequence
        label = data.iloc[i + sequence_length,0]  # Label for the next record
        date = data.iloc[i + sequence_length].name
        X.append(seq)
        y.append(label)
        dates.append(date)
    return np.array(X), np.array(y), dates

# Define LSTM Model
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers):
        super(StockLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, 1)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for sequences, targets in train_loader:
            optimizer.zero_grad()
            output = model(sequences)
            targets = targets.view_as(output)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1} Loss {loss.item():.4f}')

# def evaluate_model(model, test_loader, test_dates):
#     model.eval()
#     predictions, actuals, prediction_dates = [], [], []
#     test_losses = []
#     with torch.no_grad():
#         for sequences, labels in test_loader:
#             output = model(sequences)
#             loss = criterion(output, labels.unsqueeze(1))
#             test_losses.append(loss.item())
#             predictions.extend(output.view(-1).tolist())
#             actuals.extend(labels.tolist())
#             prediction_dates.extend(test_dates[i * sequences.size(0):i * sequences.size(0) + sequences.size(0)])
#         average_loss = np.mean(test_losses)
#         print(stock, f'Average Loss on Test Set: {average_loss:.4f}') 
#     return predictions, actuals, prediction_dates, average_loss

def evaluate_model(model, test_loader, test_dates, scaler):
    model.eval()
    predictions, actuals, prediction_dates, losses = [], [], [], []
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for batch_index, (sequences, labels) in enumerate(test_loader):
            output = model(sequences)
            loss = criterion(output.view(-1), labels)  # Make sure the dimensions align for loss calculation

            # Collect batch outputs for predictions, actuals, and dates
            pred_scaled = output.view(-1).tolist()

            # Convert predictions to a NumPy array and reshape for inverse transformation
            pred_scaled = np.array(pred_scaled).reshape(-1, 1)

            # Create a dummy array that matches the scaler's expected input shape
            dummy_array = np.zeros((pred_scaled.shape[0], scaler.min_.shape[0]))
            dummy_array[:, 0] = pred_scaled.flatten()  # Assuming the target is the first column

            # Inverse transform and only take the first column (assuming the first column is the target)
            inv_transformed_preds = scaler.inverse_transform(dummy_array)[:, 0]
            predictions.extend(inv_transformed_preds.flatten())
            
            # Add actuals to the list (ensure they are not scaled)
            actuals.extend(labels.tolist())
            
            # Calculate the start and end index for the dates in this batch
            start_index = batch_index * test_loader.batch_size
            end_index = start_index + sequences.size(0)
            prediction_dates.extend(test_dates[start_index:end_index])

            # Accumulate loss and count for averaging
            total_loss += loss.item() * sequences.size(0)  # Total loss for averaging later
            total_count += sequences.size(0)  # Total number of items

    average_loss = total_loss / total_count if total_count != 0 else 0
    return predictions, actuals, prediction_dates, average_loss

# Hyperparameters
input_size = 4  # Number of features: stock price, treasry rate, CPI, unemployment
hidden_layer_size = 50
num_layers = 3
seq_length = 30 # 30 days include change of CPI and employment
batch_size = 64 #64
num_epochs = 10 #10
learning_rate = 0.001

# Load and preprocess data
data_df,industry_data,market_cap_data= load_data("sp500_stock_prices.csv","sp500_tickers_and_industries.csv",
                                                 "sp500_type_market_cap.csv","macro_data.csv")
# data_df = data_df[['Date','A','AAPL','^TNX','CPI Ratio','Unemployment Rate']]
data_df.set_index('Date', inplace=True)


## industry experiment ##
# industry_list = industry_data['GICS Sector'].unique()
# loss_dict = {}
# stock_loss_dict={}
# loss = 0
# returns_df = pd.DataFrame()

# for industry in industry_list:
#     stock_list = list(industry_data[industry_data['GICS Sector'] == industry]['Symbol'])
#     # stock_list.extend(['^TNX','CPI Ratio','Unemployment Rate'])
#     loss_list = []
#     for stock in stock_list:       
#         if stock in data_df.columns:
#             print(industry, stock)
#             cols = [stock,'^TNX','CPI Ratio','Unemployment Rate']
#             stock_df = data_df[cols]

#             #scale the columns
#             stock_df,scaler = scale_data(stock_df)
#             # print("Center values (median):", scaler.data_min_)
#             # print("Center values (median):", scaler.data_max_)
#             # print("Scale values (IQR):", scaler.scale_)
#             #split training and test set
#             train_size = int(0.9 * len(stock_df))
#             test_size = len(stock_df) - train_size
#             train_df, test_df = stock_df.iloc[0:train_size], stock_df.iloc[train_size:len(stock_df)]
#             #create model
#             model = StockLSTM(input_size, hidden_layer_size, num_layers)
#             criterion = nn.MSELoss()
#             optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#             #create sequence
#             train_sequences, train_labels, train_dates = create_sequences(train_df, seq_length)
#             test_sequences, test_labels, test_dates = create_sequences(test_df, seq_length)

#             # DataLoader for the train set
#             train_dataset = TensorDataset(torch.tensor(train_sequences).float(), torch.tensor(train_labels).float())
#             train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

#             # DataLoader for the test set
#             test_dataset = TensorDataset(torch.tensor(test_sequences).float(), torch.tensor(test_labels).float())
#             test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

#             # Training the model
#             train_model(model, train_loader, criterion, optimizer, num_epochs)

#             #make predictions
#             predictions, actuals, prediction_dates, average_loss = evaluate_model(model, test_loader, test_dates, scaler)

#             #returns to df
#             col_actual_str = stock+'_actual'  
#             col_pred_str = stock+'_pred'  
#             df = pd.DataFrame({
#                     'Date': prediction_dates,
#                     col_pred_str: predictions,
#                     col_actual_str: actuals
#                     })
#             if returns_df.empty:
#                 returns_df = df
#             else:
#                 returns_df = pd.merge(returns_df, df, on='Date',how='outer')

#             stock_loss_dict[stock] = average_loss
#             loss_list.append(average_loss)

#     loss = np.mean(loss_list)  
#     loss_dict[industry] = loss

# # Convert to DataFrame
# stock_loss_dict_df = pd.DataFrame(stock_loss_dict,index=[0])
# loss_dict_df = pd.DataFrame(loss_dict,index=[0])

# # Save to CSV
# stock_loss_dict_df.to_csv('stock_loss_dict.csv', index=False)
# loss_dict_df.to_csv('loss_dict.csv', index=False)
# returns_df.to_csv("returns_df.csv",index=False)


## Cap experiment ##
market_cap_data['Percentile'] = market_cap_data['Market Cap'].rank(pct=True)
# def categorize_stocks_cap(data_df):
#     large_cap = data_df[data_df['Market Cap'] > 10_000_000_000]['Symbol'].tolist()
#     mid_cap = data_df[(data_df['Market Cap'] <= 10_000_000_000) & (data_df['Market Cap'] > 2_000_000_000)]['Symbol'].tolist()
#     small_cap = data_df[data_df['Market Cap'] <= 2_000_000_000]['Symbol'].tolist()
#     return {'Large Cap': large_cap, 'Mid Cap': mid_cap, 'Small Cap': small_cap}
# Categorize the stocks
# cap_categories = categorize_stocks_cap(market_cap_data)
type_list = market_cap_data['Type_Cap_Category'].unique()
loss_dict = {}
stock_loss_dict={}
loss = 0
returns_df = pd.DataFrame()
# Iterate through each category
for category in type_list:
    stock_list = list(market_cap_data[market_cap_data.Type_Cap_Category==category]['Ticker'])
    loss_list = []
    for stock in stock_list:       
        if stock in data_df.columns:
            print(category, stock)
            cols = [stock, '^TNX', 'CPI Ratio', 'Unemployment Rate']
            stock_df = data_df[cols]

            # Scale the columns
            stock_df, scaler = scale_data(stock_df)

            # Split training and test set
            train_size = int(0.9 * len(stock_df))
            test_size = len(stock_df) - train_size
            train_df, test_df = stock_df.iloc[0:train_size], stock_df.iloc[train_size:len(stock_df)]

            # Create model
            model = StockLSTM(input_size, hidden_layer_size, num_layers)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Create sequence
            train_sequences, train_labels, train_dates = create_sequences(train_df, seq_length)
            test_sequences, test_labels, test_dates = create_sequences(test_df, seq_length)

            # DataLoader for the train set
            train_dataset = TensorDataset(torch.tensor(train_sequences).float(), torch.tensor(train_labels).float())
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

            # DataLoader for the test set
            test_dataset = TensorDataset(torch.tensor(test_sequences).float(), torch.tensor(test_labels).float())
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

            # Training the model
            train_model(model, train_loader, criterion, optimizer, num_epochs)

            # Make predictions
            predictions, actuals, prediction_dates, average_loss = evaluate_model(model, test_loader, test_dates, scaler)

            # Returns to df
            col_actual_str = stock + '_actual'  
            col_pred_str = stock + '_pred'  
            df = pd.DataFrame({
                'Date': prediction_dates,
                col_pred_str: predictions,
                col_actual_str: actuals
            })
            if returns_df.empty:
                returns_df = df
            else:
                returns_df = pd.merge(returns_df, df, on='Date', how='outer')

            stock_loss_dict[stock] = average_loss
            loss_list.append(average_loss)

    loss = np.mean(loss_list)  
    loss_dict[category] = loss

# Convert to DataFrame
stock_loss_dict_df = pd.DataFrame(stock_loss_dict,index=[0])
loss_dict_df = pd.DataFrame(loss_dict,index=[0])

# Save to CSV
stock_loss_dict_df.to_csv('cap_stock_loss_dict.csv', index=False)
loss_dict_df.to_csv('cap_loss_dict.csv', index=False)
returns_df.to_csv("cap_returns_df.csv",index=False)