#import os
#os.chdir(r"C:\Users\Ajinkya\Documents\Options-Pricing-Project")
#exec(open('american-pricing-draft.py').read())

import QuantLib as ql

today = ql.Date().todaysDate()
riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.05, ql.Actual365Fixed()))
dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.01, ql.Actual365Fixed()))
volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), 0.1, ql.Actual365Fixed()))
initialValue = ql.QuoteHandle(ql.SimpleQuote(150))
process = ql.BlackScholesMertonProcess(initialValue, dividendTS, riskFreeTS, volatility)

option_type = ql.Option.Call
payoff = ql.PlainVanillaPayoff(option_type, 65)
end_date = today + 180
am_exercise = ql.AmericanExercise(today, end_date)
american_option = ql.VanillaOption(payoff, am_exercise)

xGrid = 200
tGrid = 2000
engine = ql.FdBlackScholesVanillaEngine(process, tGrid, xGrid)
american_option.setPricingEngine(engine)
print(american_option.NPV())