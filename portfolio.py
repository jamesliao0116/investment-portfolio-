import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NUM_STOCK = 23 #number of maximum stocks in the portfolio
EW_strategy = True
Sharpe_strategy = True
MinRisk_strategy = True
Plot_MonteCarlo = False
Plot_Sharpe_Point = False
Plot_MinRisk_Point = False

# Function that generate cumulative return plot
def cumulative_returns_plot(name_list):
    for name in name_list:
        CumulativeReturns = ((1+ETFreturns[name]).cumprod()-1)
        print(f'{name} Cumulative Returns: {CumulativeReturns[-1]}')
        CumulativeReturns.plot(label=name)
    plt.legend()
    plt.show()

# 1. Read the raw data
prices = pd.read_csv('ETF_prices.csv', parse_dates=['Date'], index_col='Date')
column_list = prices.columns.values.tolist()
ETFreturns = prices.pct_change()
ETF_return = ETFreturns.copy()

# 2. Set up equal weight strategy and record the result (cumulative returns and annual volatility)
if EW_strategy:
    portfolio_weights_ew = np.repeat(1/NUM_STOCK, NUM_STOCK)
    ETFreturns['Portfolio_EW'] = ETF_return.mul(portfolio_weights_ew, axis=1).sum(axis=1)
    cov_mat = ETF_return.cov()
    cov_mat_annual = cov_mat * 252
    EW_volatility = np.sqrt(np.dot(portfolio_weights_ew.T, np.dot(cov_mat_annual, portfolio_weights_ew)))

# 3. Set up Monte-Carlo simulation configuration
if Sharpe_strategy or MinRisk_strategy:
    number = 10000
    samples = np.empty((number, 25)) #23 weights for each productm 1 returns and 1 volatility 
    np.random.seed(123)

    for i in range(number):
        random_num = np.random.random(NUM_STOCK)
        random_weight = random_num / np.sum(random_num)
        
        # mean annualized return
        mean_return = ETF_return.mul(random_weight, axis=1).sum(axis=1).mean()
        annual_return = (1 + mean_return)**252 - 1

        # annualized volatility
        random_volatility = np.sqrt(np.dot(random_weight.T, np.dot(cov_mat_annual, random_weight)))

        # put the results into the record set
        samples[i][:23] = random_weight
        samples[i][23] = annual_return
        samples[i][24] = random_volatility

        RandomPortfolios = pd.DataFrame(samples)
        RandomPortfolios.columns = [column + "_weight" for column in column_list]  + ['Returns', 'Volatility']
        
    if Plot_MonteCarlo:
        RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
        plt.show()

# 4. Set up sharpe ratio strategy configuration and find the result
if Sharpe_strategy:
    risk_free = 0
    # calculate the sharpe ratio for each data
    RandomPortfolios['Sharpe'] = (RandomPortfolios.Returns - risk_free) / RandomPortfolios.Volatility
    # find the index of data that have biggest sharpe ratio
    max_sharpe_index = RandomPortfolios.Sharpe.idxmax()

    # retrieve the weights that generate biggest sharpe ratio
    MSR_weights = np.array(RandomPortfolios.iloc[max_sharpe_index, 0:NUM_STOCK])

    # Get return of sharpe ratio strategy
    ETFreturns['Portfolio_Sharpe'] = ETF_return.mul(MSR_weights, axis=1).sum(axis=1)
    if Plot_Sharpe_Point:
        RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
        x = RandomPortfolios.loc[max_sharpe_index,'Volatility']
        y = RandomPortfolios.loc[max_sharpe_index,'Returns']
        plt.scatter(x, y, color='red')   
        plt.show()

# 5. Set up minimum risk strategy configuration and find the result
if MinRisk_strategy:
    min_risk_index = RandomPortfolios.Volatility.idxmin()
    GMV_weights = np.array(RandomPortfolios.iloc[min_risk_index, 0:NUM_STOCK])
    ETFreturns['Portfolio_MinRisk'] = ETF_return.mul(GMV_weights, axis=1).sum(axis=1)
    if Plot_Sharpe_Point:
        RandomPortfolios.plot('Volatility', 'Returns', kind='scatter', alpha=0.3)
        x = RandomPortfolios.loc[min_risk_index,'Volatility']
        y = RandomPortfolios.loc[min_risk_index,'Returns']
        plt.scatter(x, y, color='red')   
        plt.show()

# 6. Display the result
output_list = []
if EW_strategy:
    output_list.append('Portfolio_EW')
if Sharpe_strategy:
    output_list.append('Portfolio_Sharpe')
if MinRisk_strategy:
    output_list.append('Portfolio_MinRisk')
cumulative_returns_plot(output_list)