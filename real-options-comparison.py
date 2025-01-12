#import os
#os.chdir(r"C:\Users\Ajinkya\Documents\Options-Pricing-Project")
#exec(open('real-options-comparison.py').read())

#print(df.columns)

#Imports
import pandas as pd
import QuantLib as ql
import numpy as np

#Function to calculate price of American options
def amer_options_price(spot, strike, mat_time, int_rate, vol):
    #Define Black-Scholes-Merton process
    today = ql.Date().todaysDate()
    riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, int_rate, ql.Actual365Fixed()))
    dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    initialValue = ql.QuoteHandle(ql.SimpleQuote(spot))
    process = ql.BlackScholesMertonProcess(initialValue, dividendTS, riskFreeTS, volatility)

    #Define the option
    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    end_date = today + int(365*mat_time)
    am_exercise = ql.AmericanExercise(today, end_date)
    american_option = ql.VanillaOption(payoff, am_exercise)

    #Define the pricing engine
    xGrid = 200
    tGrid = 2000
    engine = ql.FdBlackScholesVanillaEngine(process, tGrid, xGrid)
    american_option.setPricingEngine(engine)

    return np.float64(american_option.NPV())

#Read CSV file into DataFrame
dat_in = pd.read_csv('AAPL-Options-Dataset/aapl_2021_2023.csv')

#Get all quotes for a selected day (latest day of dataset)
dat_sel = dat_in[dat_in['QUOTE_DATE'] == '2023-03-31']

#Filter by options expiring on a given date roughly six months after
dat_sel = dat_sel[dat_sel['EXPIRE_DATE'] == '2023-09-15']

#Go through all call contracts in selected data
for call_cont in dat_sel.itertuples():
    #Calculate the theoretical American options price
    spot = call_cont.UNDERLYING_LAST
    strike = call_cont.STRIKE
    mat_time = 0.5 
    int_rate = 0.05
    vol = 0.1
    amer_pr = amer_options_price(spot, strike, mat_time, int_rate, vol)
    act_pr = call_cont.C_LAST
    diff_pr = 100*(act_pr - amer_pr)/amer_pr

    #Print out the values
    print("Strike: %3.2f, Theoretical: %3.2f, Actual: %3.2f, Diff: %2.2f" % (strike, amer_pr, act_pr, diff_pr))