#!/usr/bin/env python


# Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Settings to produce nice plots in a Jupyter notebook
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading in the data
stock_data = pd.read_csv("datasets/stock_data.csv",parse_dates=['Date'],index_col=['Date']).dropna()
benchmark_data = pd.read_csv("datasets/benchmark_data.csv",parse_dates=['Date'],index_col=['Date']).dropna()


get_ipython().run_cell_magic('nose', '', '\ndef test_benchmark_data():\n    assert isinstance(benchmark_data, pd.core.frame.DataFrame), \\\n        \'Did you import the benchmark_data as a DataFrame?\'\n\ndef test_stock_data():\n    assert isinstance(stock_data, pd.core.frame.DataFrame), \\\n        \'Did you import the stock_data as a DataFrame?\'\n\ndef test_benchmark_index():\n    assert isinstance(benchmark_data.index, pd.core.indexes.datetimes.DatetimeIndex), \\\n        "Did you set the \'Date\' column as Index for the benchmark_data?"\n\ndef test_stock_index():\n    assert isinstance(stock_data.index, pd.core.indexes.datetimes.DatetimeIndex), \\\n        "Did you set the \'Date\' column as Index for the stock_data?"\n\ndef test_stock_data_shape():\n    assert stock_data.shape == (252, 2), \\\n        "Did you use .dropna() on the stock_data?"\n\ndef test_stock_benchmark_shape():\n    assert benchmark_data.shape == (252, 1), \\\n        "Did you use .dropna() on the benchmark_data?"\n    ')


# ## 2. A first glance at the data
# <p>Let's take a look the data to find out how many observations and variables we have at our disposal.</p>

# In[185]:


# Display summary for stock_data
print('Stocks\n')
stock_data.info()
print(stock_data.head())


# Display summary for benchmark_data
print('\nBenchmarks\n')

print(benchmark_data.head())


# In[186]:


get_ipython().run_cell_magic('nose', '', '\ndef test_nothing():\n    pass')


# ## 3. Plot & summarize daily prices for Amazon and Facebook
# <p>Before we compare an investment in either Facebook or Amazon with the index of the 500 largest companies in the US, let's visualize the data, so we better understand what we're dealing with.</p>

# In[187]:


# visualize the stock_data

stock_data.plot(title='Stock Data',subplots=True)


# summarize the stock_data

print(stock_data.info())


# In[188]:


get_ipython().run_cell_magic('nose', '', '\ndef test_nothing():\n    pass')


# ## 4. Visualize & summarize daily values for the S&P 500
# <p>Let's also take a closer look at the value of the S&amp;P 500, our benchmark.</p>

# In[189]:


# plot the benchmark_data

benchmark_data.plot(title='Benchmark Data',subplots=True)


# summarize the benchmark_data

print(benchmark_data.describe())


# In[190]:


get_ipython().run_cell_magic('nose', '', '\ndef test_nothing():\n    pass')


# ## 5. The inputs for the Sharpe Ratio: Starting with Daily Stock Returns
# <p>The Sharpe Ratio uses the difference in returns between the two investment opportunities under consideration.</p>
# <p>However, our data show the historical value of each investment, not the return. To calculate the return, we need to calculate the percentage change in value from one day to the next. We'll also take a look at the summary statistics because these will become our inputs as we calculate the Sharpe Ratio. Can you already guess the result?</p>

# In[191]:


# calculate daily stock_data returns
stock_returns = stock_data.pct_change()

# plot the daily returns

stock_returns.plot()

# summarize the daily returns

stock_returns.describe()


# In[192]:


get_ipython().run_cell_magic('nose', '', "\ndef test_stock_returns():\n    assert stock_returns.equals(stock_data.pct_change()), \\\n    'Did you use pct_change()?'")


# ## 6. Daily S&P 500 returns
# <p>For the S&amp;P 500, calculating daily returns works just the same way, we just need to make sure we select it as a <code>Series</code> using single brackets <code>[]</code> and not as a <code>DataFrame</code> to facilitate the calculations in the next step.</p>

# In[193]:


# calculate daily benchmark_data returns

sp_returns = benchmark_data['S&P 500'].pct_change()

# plot the daily returns

sp_returns.plot()

# summarize the daily returns

sp_returns.describe()


# In[194]:


get_ipython().run_cell_magic('nose', '', "\ndef test_sp_returns():\n    assert sp_returns.equals(benchmark_data['S&P 500'].pct_change()), \\\n    'Did you use pct_change()?'")


# ## 7. Calculating Excess Returns for Amazon and Facebook vs. S&P 500
# <p>Next, we need to calculate the relative performance of stocks vs. the S&amp;P 500 benchmark. This is calculated as the difference in returns between <code>stock_returns</code> and <code>sp_returns</code> for each day.</p>

# In[195]:


# calculate the difference in daily returns
excess_returns = stock_returns.sub(sp_returns,axis=0)

# plot the excess_returns

excess_returns.plot()

# summarize the excess_returns

sp_returns.describe()


# In[196]:


get_ipython().run_cell_magic('nose', '', "\ndef test_excess_returns():\n    assert excess_returns.equals(stock_returns.sub(sp_returns, axis=0)), \\\n    'Did you use .sub()?'")


# ## 8. The Sharpe Ratio, Step 1: The Average Difference in Daily Returns Stocks vs S&P 500
# <p>Now we can finally start computing the Sharpe Ratio. First we need to calculate the average of the <code>excess_returns</code>. This tells us how much more or less the investment yields per day compared to the benchmark.</p>

# In[197]:


# calculate the mean of excess_returns 

avg_excess_return = excess_returns.mean()

# plot avg_excess_returns

avg_excess_return.plot.bar(title='Mean of the Return Difference')


# In[198]:


get_ipython().run_cell_magic('nose', '', "\ndef test_avg_excess_return():\n    assert avg_excess_return.equals(excess_returns.mean()), \\\n    'Did you use .mean()?'")


# ## 9. The Sharpe Ratio, Step 2: Standard Deviation of the Return Difference
# <p>It looks like there was quite a bit of a difference between average daily returns for Amazon and Facebook.</p>
# <p>Next, we calculate the standard deviation of the <code>excess_returns</code>. This shows us the amount of risk an investment in the stocks implies as compared to an investment in the S&amp;P 500.</p>

# In[199]:


# calculate the standard deviations
sd_excess_return = excess_returns.std()

# plot the standard deviations

sd_excess_return.plot.bar(title='Standard Deviation of the Return Difference')


# In[200]:


get_ipython().run_cell_magic('nose', '', "\ndef test_sd_excess():\n    assert sd_excess_return.equals(excess_returns.std()), \\\n    'Did you use .std() on excess_returns?'")


# ## 10. Putting it all together
# <p>Now we just need to compute the ratio of <code>avg_excess_returns</code> and <code>sd_excess_returns</code>. The result is now finally the <em>Sharpe ratio</em> and indicates how much more (or less) return the investment opportunity under consideration yields per unit of risk.</p>
# <p>The Sharpe Ratio is often <em>annualized</em> by multiplying it by the square root of the number of periods. We have used daily data as input, so we'll use the square root of the number of trading days (5 days, 52 weeks, minus a few holidays): √252</p>

# In[201]:


# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plot the annualized sharpe ratio

annual_sharpe_ratio.plot.bar(title='Annualized Sharpe Ratio: Stocks vs S&P 500')


# In[202]:


get_ipython().run_cell_magic('nose', '', "\ndef test_daily_sharpe():\n    assert daily_sharpe_ratio.equals(avg_excess_return.div(sd_excess_return)), \\\n    'Did you use .div() avg_excess_return and sd_excess_return?'\n    \ndef test_annual_factor():\n    assert annual_factor == np.sqrt(252), 'Did you apply np.sqrt() to, number_of_trading_days?'\n    \ndef test_annual_sharpe():\n    assert annual_sharpe_ratio.equals(daily_sharpe_ratio.mul(annual_factor)), 'Did you use .mul() with daily_sharpe_ratio and annual_factor?'")


# ## 11. Conclusion
# <p>Given the two Sharpe ratios, which investment should we go for? In 2016, Amazon had a Sharpe ratio twice as high as Facebook. This means that an investment in Amazon returned twice as much compared to the S&amp;P 500 for each unit of risk an investor would have assumed. In other words, in risk-adjusted terms, the investment in Amazon would have been more attractive.</p>
# <p>This difference was mostly driven by differences in return rather than risk between Amazon and Facebook. The risk of choosing Amazon over FB (as measured by the standard deviation) was only slightly higher so that the higher Sharpe ratio for Amazon ends up higher mainly due to the higher average daily returns for Amazon. </p>
# <p>When faced with investment alternatives that offer both different returns and risks, the Sharpe Ratio helps to make a decision by adjusting the returns by the differences in risk and allows an investor to compare investment opportunities on equal terms, that is, on an 'apples-to-apples' basis.</p>

# In[203]:


# Uncomment your choice.
buy_amazon = True
# buy_facebook = True


# In[204]:


get_ipython().run_cell_magic('nose', '', "\ndef test_decision():\n    assert 'buy_amazon' in globals() and buy_amazon == True, \\\n    'Which stock has the higher Sharpe Ratio'")

