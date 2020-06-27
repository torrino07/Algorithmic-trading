######## Libraries ############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
###############################

######## Global variables #####
global col
col = 'Adj Close'
########## Filpaths ###########
filepath = glob.glob('/Users/giuliano/Programming/Python/Data/*.csv')

###############################

########### Methods ###########

def reader(filepath):
	data = pd.read_csv(filepath, delimiter=',')
	data = data.set_index(pd.DatetimeIndex(data['Date'].values))
	return data

def returns(filepath):
	Index = reader(filepath)
	inReturns = Index[col].diff(1)
	return inReturns

def averageGain(filepath,k):
	inReturns = returns(filepath)
	up = inReturns.copy()
	up[up<0]=0
	inAverageGain = up.rolling(window = k-1).mean()
	return inAverageGain	

def averageLoss(filepath,k):
	inReturns = returns(filepath)
	down = inReturns.copy()
	down[down>0]=0
	inAverageLoss = abs(down.rolling(window = k-1).mean())
	return inAverageLoss

def RS(filepath,k):
	inAverageGain = averageGain(filepath,k)
	inAverageLoss = averageLoss(filepath,k)
	inRS = inAverageGain/inAverageLoss
	return inRS

def RSI(filepath,k):
	inRS = RS(filepath,k)
	inRSI = 100 - (100/(1+inRS))
	return inRSI

def HighestHigh(filepath,k):
	Index = reader(filepath)
	High = reader(filepath)['High']
	n = len(High)
	HighestHigh = []
	for i in range(n-k+1):
		HighestHigh.append(max(High[i:i+k]))

	HighestHighK = DF(Index,HighestHigh,k)
	return HighestHighK

def LowestLow(filepath,k):
	Index = reader(filepath)
	Low = reader(filepath)['Low']
	n = len(Low)
	LowestLow = []
	for i in range(n-k+1):
		LowestLow.append(min(Low[i:i+k]))

	LowestLowK = DF(Index,LowestLow,k)
	return LowestLowK

def maxRSI(filepath,k):
	Index = reader(filepath)
	inRSI = RSI(filepath,k+1)
	n = len(inRSI)
	maxRSIk = []
	for i in range(n-k+1):
		maxRSIk.append(max(inRSI[i:i+k]))

	maxRSIK = DF(Index,maxRSIk,k)
	return maxRSIK

def minRSI(filepath,k):
	Index = reader(filepath)
	inRSI = RSI(filepath,k+1)
	n = len(inRSI)
	minRSIk = []
	for i in range(n-k+1):
		minRSIk.append(min(inRSI[i:i+k]))

	minRSIK = DF(Index,minRSIk,k)
	return minRSIK

def Stochastic(filepath,k):
	Index = reader(filepath)
	H = HighestHigh(filepath,k)[0]
	L = LowestLow(filepath,k)[0]
	C = reader(filepath)['Adj Close'][k-1:]
	C = DF(Index,C,k)['Adj Close']
	
	K = []
	for c,h,l in zip(C,H,L):
		K.append(((c-l)/(h-l))*100)
	K = DF(Index,K,0)

	return K

def StochasticRSI(filepath,k):
	Index = reader(filepath)
	inRSImax = maxRSI(filepath,k)[0]
	inRSImin = minRSI(filepath,k)[0]
	inRSI = RSI(filepath,k+1)


	K = []
	for r,l,h in zip(inRSI,inRSImin,inRSImax):
		K.append(((r-l)/(h-l))*100)

	K = DF(Index,K,0)

	return K

def smaStochastic(filepath,k1,k2):
	Index = reader(filepath)
	inStochastic = Stochastic(filepath,k2)[0][k2:]

	n = len(inStochastic)
	SMA = []
	for i in range(n-k1+1):
		SMA.append((sum(inStochastic[i:i+k1])/k1))
	
	K = DF(Index,SMA,k1+k2)
	return K

def smaStochasticRSI(filepath,k1,k2):
	Index = reader(filepath)
	inStochasticRSI = StochasticRSI(filepath,k2)[0][k2+k2:]

	n = len(inStochasticRSI)
	SMA = []
	for i in range(n-k1+1):
		SMA.append((sum(inStochasticRSI[i:i+k1])/k1))
	
	K = DF(Index,SMA,k1+k2+k2)
	return K

def EMA(filepath,k):
	w1 = (2/(k + 1))
	w2 = 1 - w1
	Index = reader(filepath)
	SubPrice = reader(filepath)['Adj Close'][:k]
	PriceSup = reader(filepath)['Adj Close'][k:]
	muS = sum(SubPrice)/k
	EMA = [muS]
	for i,q in zip(PriceSup,EMA):
		EMA.append(((i*w1)+(q*w2)))

	K = DF(Index,EMA,k)
	return K

def MACD(filepath,k1,k2):
	Index = reader(filepath)
	inEMAk1 = EMA(filepath,k1)[0]
	inEMAk2 = EMA(filepath,k2)[0]

	MACD = []
	for i,k in zip(inEMAk1,inEMAk2):
		MACD.append(i-k)

	K = DF(Index,MACD,0)
	return K

def Signal(filepath,k1,k2,k3):
	w1 = (2/(k3 + 1))
	w2 = 1 - w1
	Index = reader(filepath)
	SubPrice = MACD(filepath,k1,k2)[0][k2-1:][:k3]
	PriceSup = MACD(filepath,k1,k2)[0][k2-1:][k3:]
	muS = sum(SubPrice)/k3
	EMA = [muS]
	for i,k in zip(PriceSup,EMA):
		EMA.append(((i*w1)+(k*w2)))

	K = DF(Index,EMA,k2+k3-1)
	return K

def MACDHist(filepath,k1,k2,k3):
	Index = reader(filepath)
	inMACD = MACD(filepath,k1,k2)[0]
	inSignal = Signal(filepath,k1,k2,k3)[0]
	histo = []
	for i,k in zip(inMACD,inSignal):
		histo.append(i-k)

	K = DF(Index,histo,0)
	return K
	

def DF(Index,col,k):
	n = len(Index)
	nan = pd.DataFrame([np.nan for i in range(0)] for i in range(k-1))
	df = pd.DataFrame(col)
	doc = nan.append(df)
	doc.reset_index(drop=True, inplace=True)
	date = pd.Series(Index['Date'].values)

	doc.insert(0,'Date',date,True)
	doc.set_index(doc['Date'], inplace=True)
	del doc['Date']

	return doc

def DFs(filepath):
	inIndex = reader(filepath)['Adj Close']

	inReturns = returns(filepath)
	inRSI = RSI(filepath,k=14)

	inStochastic = Stochastic(filepath,k=14)
	inStochasticRSI = StochasticRSI(filepath,k=14)
	insmaStochastic = smaStochastic(filepath,k1=3,k2=14)
	insmaStochasticRSI = smaStochasticRSI(filepath,k1=3,k2=14)

	inEMA12 = EMA(filepath,k=12)
	inEMA26 = EMA(filepath,k=26)

	inMACD = MACD(filepath,k1=12,k2=26)
	inSignal = Signal(filepath,k1=12,k2=26,k3=9)
	inMACDHisto = MACDHist(filepath,k1=12,k2=26,k3=9)

	NDF = pd.DataFrame()
	NDF['Adj Close'] = inIndex

	NDF['Returns'] = inReturns
	NDF['RSI'] = inRSI

	NDF['Stochastic 14'] = inStochastic[0]
	NDF['SMA 3'] = insmaStochastic[0]

	NDF['Stochastic RSI 14'] = inStochasticRSI[0]
	NDF['SMA RSI 3'] = insmaStochasticRSI[0]

	NDF['EMA 12'] = inEMA12[0]
	NDF['EMA 26'] = inEMA26[0]

	NDF['MACD 12 26'] = inMACD[0]
	NDF['Signal'] = inSignal[0]
	NDF['MACD Histogram'] = inMACDHisto[0]

	return NDF

def charts(NDFs,n):

	plt.suptitle(n +'($)')
	plt.subplot(6, 1, 1, facecolor='black')
	plt.plot(NDFs.iloc[:,0],linewidth=0.60)
	plt.plot(NDFs.iloc[:,7],linewidth=0.60,)
	plt.plot(NDFs.iloc[:,8],linewidth=0.60,)
	plt.grid(linestyle='-', linewidth='0.15', color='white')
	plt.legend(loc=2,prop={'size': 5})

	plt.subplot(6, 1, 2, facecolor='black')
	plt.plot(NDFs.iloc[:,1],linewidth=0.60,)
	plt.grid(linestyle='-', linewidth='0.15', color='white')
	plt.legend(loc=2,prop={'size': 5})

	plt.subplot(6, 1, 3, facecolor='black')
	plt.plot(NDFs.iloc[:,2],linewidth=0.60,color='mediumblue')

	s = [20,80]
	c = ['orangered','orangered']
	for h, k in zip(s,c):
		plt.axhline(h,linestyle='--',linewidth=0.65, alpha = 0.5, color=k)
		plt.fill_between(NDFs.index,20,80,color='lavenderblush',alpha=0.5)
		plt.grid(linestyle='-', linewidth='0.15', color='white')
		plt.legend(loc=2,prop={'size': 5})

	plt.subplot(6, 1, 4, facecolor='black')
	plt.plot(NDFs.iloc[:,3],linewidth=0.60,color='mediumblue')
	plt.plot(NDFs.iloc[:,4],linewidth=0.60,color='orangered')
	s = [20,80]
	c = ['orangered','orangered']
	for h, k in zip(s,c):
		plt.axhline(h,linestyle='--',linewidth=0.65, alpha = 0.5, color=k)
		plt.fill_between(NDFs.index,20,80,color='lavenderblush',alpha=0.5)
		plt.grid(linestyle='-', linewidth='0.15', color='white')
		plt.legend(loc=2,prop={'size': 5})

	plt.subplot(6, 1, 5, facecolor='black')
	plt.plot(NDFs.iloc[:,5],linewidth=0.60,color='mediumblue')
	plt.plot(NDFs.iloc[:,6],linewidth=0.60,color='orangered')
	s = [20,80]
	c = ['orangered','orangered']
	for h, k in zip(s,c):
		plt.axhline(h,linestyle='--',linewidth=0.65, alpha = 0.5, color=k)
		plt.fill_between(NDFs.index,20,80,color='lavenderblush',alpha=0.5)
		plt.grid(linestyle='-', linewidth='0.15', color='white')
		plt.legend(loc=2,prop={'size': 5})

	plt.subplot(6, 1, 6, facecolor='black')
	plt.plot(NDFs.iloc[:,9],linewidth=0.60, color='mediumblue')
	plt.plot(NDFs.iloc[:,10],linewidth=0.60, color='orangered')
	plt.fill_between(NDFs.index, NDFs.iloc[:,11], alpha='0.5')
	plt.fill_between(NDFs.index, NDFs.iloc[:,11], where=(NDFs.iloc[:,11]>=0),color='green',alpha=0.5)
	plt.fill_between(NDFs.index, NDFs.iloc[:,11], where=(NDFs.iloc[:,11]<0),color='red',alpha=0.5,interpolate=True)
	plt.grid(linestyle='-', linewidth='0.15', color='white')
	plt.legend(loc=2,prop={'size': 5})
	plt.show()

###############################

######## Def Main #############
def main(filepath):

	# Magic numbers #

    # Initailization
   
    assets = ['ETH-USD','BTC-USD','XRP-USD','XLM-USD','LTC-USD']
    for i,l in zip(filepath,assets):
    	NDFs = DFs(i)
    	charts(NDFs,l)

	# Estimation #




if __name__ == "__main__":
	main(filepath)