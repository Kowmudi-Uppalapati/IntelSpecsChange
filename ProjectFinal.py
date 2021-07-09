#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:14:24 2020

@author: kowmudiuppalapati
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
# import mplcursors


columns=['Vertical_Segment','Processor_Number','Launch_Date','Lithography','Recommended_Customer_Price','nb_of_Cores','nb_of_Threads','Processor_Base_Frequency','TDP','Intel_64_','Instruction_Set']
dataset = pd.read_csv("Intel_CPUs.csv",usecols=columns)

dataset=dataset[dataset["Launch_Date"].notnull()].reset_index(drop=True)
print("Dataset Information:")
print()
print(dataset.info())
print("--------------------------------------")
corrMatrix = dataset.corr()
print("Correlation Matrix:")
print()
# correlation_values = corrMatrix["Processor_Base_Frequency"].sort_values(ascending=False)
print(corrMatrix)
print("--------------------------------------")


Mobile = dataset[dataset['Vertical_Segment'] == 'Mobile'].reset_index(drop=True)
print()
# print(Mobile['TDP'].head())

   
Desktop = dataset[dataset['Vertical_Segment'] == 'Desktop'].reset_index(drop=True)
# print(Desktop)

Embedded = dataset[dataset['Vertical_Segment'] == 'Embedded'].reset_index(drop=True)
# print(Embedded)

Server = dataset[dataset['Vertical_Segment'] == 'Server'].reset_index(drop=True)
# print(Server)


def LaunchYear(data):
    
    try:
        year = int(data[3:])
    except:
        year = int(data[4:])
    
    if year>=0 and year <10:
        year = '200' + str(year)   
    elif year>=10 and year<19:
        year='20' + str(year)
    else:
        year = '19' + str(year)
        
    return year

Mobile=Mobile.copy()
Mobile["year"] = Mobile['Launch_Date'].apply(LaunchYear).astype(int)
print()
# print(Mobile['year'].head())

Desktop=Desktop.copy()
Desktop["year"] = Desktop['Launch_Date'].apply(LaunchYear).astype(int)
# print(Desktop['year'].unique())

Server=Server.copy()
Server["year"] = Server['Launch_Date'].apply(LaunchYear).astype(int)
# print(Server['year'].unique())

def Lithography(data):
    val = int(data[:-2])
    return val

Mobile.loc[(pd.notnull(Mobile["Lithography"])) , 'Lithography'] = Mobile.loc[(pd.notnull(Mobile["Lithography"])) , 'Lithography'].apply(Lithography)
# print(Mobile["Lithography"][64:70])

Desktop.loc[(pd.notnull(Desktop["Lithography"])) , 'Lithography'] = Desktop.loc[(pd.notnull(Desktop["Lithography"])) , 'Lithography'].apply(Lithography)
# print(Desktop["Lithography"].unique())

Server.loc[(pd.notnull(Server["Lithography"])) , 'Lithography'] = Server.loc[(pd.notnull(Server["Lithography"])) , 'Lithography'].apply(Lithography)
# print(Server["Lithography"].unique())

def TDP(data):
    data = data[:-1]
    return data

Mobile.loc[(pd.notnull(Mobile["TDP"])) , 'TDP']= Mobile.loc[(pd.notnull(Mobile["TDP"])) , 'TDP'].apply(TDP)
Mobile["TDP"] = Mobile["TDP"].astype(float)
print("TDP(mobile):")
print()
print(Mobile["TDP"])
print("--------------------------------------")

Desktop.loc[(pd.notnull(Desktop["TDP"])) , 'TDP']= Desktop.loc[(pd.notnull(Desktop["TDP"])) , 'TDP'].apply(TDP)
Desktop["TDP"] = Desktop["TDP"].astype(float)
print("TDP(Desktop):")
print()
print(Desktop["TDP"])
print("--------------------------------------")
Server.loc[(pd.notnull(Server["TDP"])) , 'TDP']= Server.loc[(pd.notnull(Server["TDP"])) , 'TDP'].apply(TDP)
Server["TDP"] = Server["TDP"].astype(float)
print("TDP(Server):")
print()
print(Server["TDP"])
print("--------------------------------------")

def clockspeed(data):
    
    if data[-3] == 'G':
        data = float(data[:-3])*1000
    else:
        data = data[:-3]
    return data

Mobile.loc[(pd.notnull(Mobile["Processor_Base_Frequency"])) , 'Processor_Base_Frequency']= Mobile.loc[(pd.notnull(Mobile["Processor_Base_Frequency"])) , 'Processor_Base_Frequency'].apply(clockspeed)
Mobile["Processor_Base_Frequency"] = Mobile["Processor_Base_Frequency"].astype(float)
print("Base Frequency(Mobile):")
print()
print(Mobile["Processor_Base_Frequency"][166:171])
print("--------------------------------------")

Desktop.loc[(pd.notnull(Desktop["Processor_Base_Frequency"])) , 'Processor_Base_Frequency']= Desktop.loc[(pd.notnull(Desktop["Processor_Base_Frequency"])) , 'Processor_Base_Frequency'].apply(clockspeed)
Desktop["Processor_Base_Frequency"] = Desktop["Processor_Base_Frequency"].astype(float)
print("Base Frequency(Desktop):")
print()
print(Desktop["Processor_Base_Frequency"])
print("--------------------------------------")

Server.loc[(pd.notnull(Server["Processor_Base_Frequency"])) , 'Processor_Base_Frequency']= Server.loc[(pd.notnull(Server["Processor_Base_Frequency"])) , 'Processor_Base_Frequency'].apply(clockspeed)
Server["Processor_Base_Frequency"] = Server["Processor_Base_Frequency"].astype(float)
print("Base Frequency(Server):")
print()
print(Server["Processor_Base_Frequency"])
print("--------------------------------------")
def price(data):
    
    data = data.replace(',','')    
    matchObj = re.match("\$([0-9]*\.[0-9]*)", data)
    
    return matchObj.group(1)

Mobile.loc[(pd.notnull(Mobile["Recommended_Customer_Price"])) , 'Recommended_Customer_Price'] = Mobile.loc[(pd.notnull(Mobile["Recommended_Customer_Price"])) , 'Recommended_Customer_Price'].apply(price)
Mobile["Recommended_Customer_Price"] = Mobile["Recommended_Customer_Price"].astype(float)
print("Customer Price(Mobile):")
print()
print(Mobile['Recommended_Customer_Price'].unique())
print("--------------------------------------")

Desktop.loc[(pd.notnull(Desktop["Recommended_Customer_Price"])) , 'Recommended_Customer_Price'] = Desktop.loc[(pd.notnull(Desktop["Recommended_Customer_Price"])) , 'Recommended_Customer_Price'].apply(price)
Desktop["Recommended_Customer_Price"] = Desktop["Recommended_Customer_Price"].astype(float)
print("Customer Price(Mobile):")
print()
print(Desktop['Recommended_Customer_Price'].unique())
print("--------------------------------------")

Server.loc[(pd.notnull(Server["Recommended_Customer_Price"])) , 'Recommended_Customer_Price'] = Server.loc[(pd.notnull(Server["Recommended_Customer_Price"])) , 'Recommended_Customer_Price'].apply(price)
Server["Recommended_Customer_Price"] = Server["Recommended_Customer_Price"].astype(float)
print("Customer Price(Mobile):")
print()
print(Server['Recommended_Customer_Price'].unique())
print("--------------------------------------")

# print(Mobile['Instruction_Set'][544:550])
# print(Mobile['Intel_64_'][544:550])

# print("correlation:")

# print(Desktop['Instruction_Set'])

# print(Server['Instruction_Set'].unique())
# print(Server.groupby(['Instruction_Set']).count())

##print(Mobile[pd.isnull(Mobile['Instruction_Set'])])
Mobile.loc[Mobile["Intel_64_"] == "Yes", 'Instruction_Set'] = "64-bit"
print("Instruction set(Mobile):")
print()
print(Mobile['Instruction_Set'][544:550]) 
print("--------------------------------------")  

##print(Desktop.loc[pd.isnull(Desktop["Instruction_Set"]) == True]  )
Desktop.loc[Desktop["Intel_64_"] == "Yes", 'Instruction_Set'] = "64-bit"
print("Instruction set(Desktop):")
print()
print(Desktop['Instruction_Set'].unique()) 
print("--------------------------------------")  

##print(Server.loc[pd.isnull(Server["Instruction_Set"]) == True])
Server.loc[Server["Intel_64_"] == "Yes", 'Instruction_Set'] = "64-bit"
print("Instruction set(Server):")
print()
print(Server['Instruction_Set'].unique())
print("--------------------------------------")

# s=Server.groupby(['Processor_Number']).size().to_frame('size')
# print(s.loc[s["size"]>1])

Mobile = Mobile.sort_values(by=['Recommended_Customer_Price']).reset_index(drop=True)
Mobile=Mobile[Mobile['Processor_Number'].isnull() | ~Mobile[Mobile['Processor_Number'].notnull()].duplicated(subset='Processor_Number',keep='first')].reset_index(drop=True)

print("Mobile")
print()
print(Mobile)
print("--------------------------------------")

Desktop = Desktop.sort_values(by=['Recommended_Customer_Price']).reset_index(drop=True)
Desktop=Desktop[Desktop['Processor_Number'].isnull() | ~Desktop[Desktop['Processor_Number'].notnull()].duplicated(subset='Processor_Number',keep='first')].reset_index(drop=True)
print("Desktop")
print()
print(Desktop)
print("--------------------------------------")

Server = Server.sort_values(by=['Recommended_Customer_Price']).reset_index(drop=True)
Server=Server[Server['Processor_Number'].isnull() | ~Server[Server['Processor_Number'].notnull()].duplicated(subset='Processor_Number',keep='first')].reset_index(drop=True)
print("Server")
print()
print(Server)
print("--------------------------------------")

"""
Frequency of releases

"""

MobileFreq = Mobile.groupby(['year']).size().to_frame('size')
print("Frequency of releases(Mobile):")
print()
print(MobileFreq)
print("--------------------------------------")

# plot1=MobileFreq1.plot(title="Frequency of releases", xticks=range(2000, 2018,2),yticks= range(2,60,5))

DesktopFreq = Desktop.groupby(['year']).size().to_frame('size')
print("Frequency of releases(Desktop):")
print()
print(DesktopFreq)
print("--------------------------------------")

ServerFreq = Server.groupby(['year']).size().to_frame('size')
print("Frequency of releases(Server):")
print()
print(ServerFreq)
print("--------------------------------------")

size=pd.concat([MobileFreq,DesktopFreq,ServerFreq],axis=1).fillna(0)
print("--------------------------------------")
##print(size)
size.plot(kind="bar")
plt.title("Frequency of releases of chips every year")
plt.xlabel("year")
plt.legend(["Mobile", "Desktop","Server"])
plt.ylabel("Count")
plt.show()


MobileFreq1 = MobileFreq.reset_index()
DesktopFreq1 = DesktopFreq.reset_index()
ServerFreq1 = ServerFreq.reset_index()

ax = plt.gca()


MobileFreq1.plot(kind='line',y='size',x='year',ax=ax)
ServerFreq1.plot(kind='line',y='size',x='year',color='orange', ax=ax)
DesktopFreq1.plot(kind='line',y='size',x='year',color='red', ax=ax)
ax.legend(["Mobile", "Server","Desktop"])
plt.xticks(np.arange(1999,2018,2))
plt.yticks(np.arange(0,100,10))
plt.title("Frequency of releases of chips every year")
plt.ylabel('count')
plt.grid()
plt.show()

"""

No. of threads
 
"""
MobileThreads=Mobile.groupby('year')["nb_of_Threads"].max().to_frame('max').reset_index()
# print(MobileThreads)

DesktopThreads=Desktop.groupby('year')["nb_of_Threads"].max().to_frame('max').reset_index()
# print(DesktopThreads)

ServerThreads=Server.groupby('year')["nb_of_Threads"].max().to_frame('max').reset_index()
# print(ServerThreads)

bx = plt.gca()

MobileThreads.plot(kind='line',x='year',y='max',ax=bx)
ServerThreads.plot(kind='line',x='year',y='max',color='orange',ax=bx)
DesktopThreads.plot(kind='line',x='year',y='max',color='red',ax=bx)
bx.legend(["Mobile", "Server","Desktop"])
plt.title("No. of threads")
plt.ylabel('Threads')
plt.xticks(np.arange(2004,2018,2))
plt.yticks(np.arange(0,100,10))
plt.grid()
plt.show()
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.suptitle('No. of threads')
# MobileThreads.plot(kind='line',x='year',y='max',ax=ax)
# ServerThreads.plot(kind='line',x='year',y='max',color='orange',ax=ax2)
# DesktopThreads.plot(kind='line',x='year',y='max',color='red',ax=ax3)

"""

No. of Cores
 
"""
MobileCores=Mobile.groupby('year')["nb_of_Cores"].max().to_frame('max').reset_index()
##print(MobileCores)

DesktopCores=Desktop.groupby('year')["nb_of_Cores"].max().to_frame('max').reset_index()
##print(DesktopCores)

ServerCores=Server.groupby('year')["nb_of_Cores"].max().to_frame('max').reset_index()
##print(ServerCores)

cx = plt.gca()

MobileCores.plot(kind='line',x='year',y='max',ax=cx)
ServerCores.plot(kind='line',x='year',y='max',color='orange',ax=cx)
DesktopCores.plot(kind='line',x='year',y='max',color='red',ax=cx)
cx.legend(["Mobile", "Server","Desktop"])
plt.title("No. of Cores")
plt.ylabel('Cores')
plt.xticks(np.arange(1999,2018,2))
# plt.yticks(np.arange(0,100,10))
plt.grid()
plt.show()

zx = plt.gca()

DesktopCores.plot(kind='line',x='year',y='max',ax=zx)
DesktopThreads.plot(kind='line',x='year',y='max',color='orange',ax=zx)
zx.legend(["Cores", "Threads"])
plt.title("No. of Cores vs Threads")
plt.ylabel('Cores')
plt.xticks(np.arange(1999,2018,2))
# plt.yticks(np.arange(0,100,10))
plt.grid()
plt.show()


"""

Demand of 32-bit and 64-bit processors change over the years

"""
MobileInstruction = Mobile.groupby(['year','Instruction_Set']).size().to_frame('size').reset_index()
MobileInstruction["year"] = "'"+((MobileInstruction["year"].astype(str)).str[2:])
MobileInstruction = MobileInstruction.pivot_table(index=['year'], columns='Instruction_Set', values='size', fill_value=0)
print(MobileInstruction.sum())
plt.rcParams['font.size'] = 10.0


plot = MobileInstruction.sum().plot(kind='pie', subplots=True, shadow = True,startangle=90,
figsize=(15,10), autopct='%1.1f%%')
plt.title('Instruction set')

dx = MobileInstruction.plot.bar(rot=0)
dx.set_title("32-bit Vs 64-bit : Mobile")
dx.set_yticks(np.arange(0,100,10))


DesktopInstruction = Desktop.groupby(['year','Instruction_Set']).size().to_frame('size').reset_index()
DesktopInstruction["year"] = "'"+((DesktopInstruction["year"].astype(str)).str[2:])
DesktopInstruction = DesktopInstruction.pivot_table(index=['year'], columns='Instruction_Set', values='size', fill_value=0)
##print(DesktopInstruction)

ex = DesktopInstruction.plot.bar(rot=0)
ex.set_title("32-bit Vs 64-bit : Desktop")
ex.set_yticks(np.arange(0,100,10))

ServerInstruction = Server.groupby(['year','Instruction_Set']).size().to_frame('size').reset_index()
ServerInstruction["year"] = "'"+((ServerInstruction["year"].astype(str)).str[2:])
ServerInstruction = ServerInstruction.pivot_table(index=['year'], columns='Instruction_Set', values='size', fill_value=0)
##print(ServerInstruction)

f = ServerInstruction.plot.bar(rot=0)
f.set_title("32-bit Vs 64-bit : Server")
f.set_yticks(np.arange(0,100,10))

"""

How has the thickness of chips change over the years.

"""

MobileLith=Mobile.groupby('year')["Lithography"].min().to_frame('Lithography')
print(MobileLith)
DesktopLith=Desktop.groupby('year')["Lithography"].min().to_frame('Lithography')
print(DesktopLith)
ServerLith=Server.groupby('year')["Lithography"].min().to_frame('Lithography')
print(ServerLith)


gx=MobileLith.plot()
ServerLith.plot(ax=gx)
DesktopLith.plot(ax=gx)
gx.set_xticks(np.arange(1999,2018,2))
gx.legend(["Mobile", "Server","Desktop"])
gx.set_title("Thickness of chips")
gx.set_ylabel('Lithography')
gx.grid()

ad=MobileLith.plot()
ad.set_xticks(np.arange(1999,2018,2))
ad.grid()
ad.set_title("Mobile")
cd=ServerLith.plot()
cd.set_xticks(np.arange(1999,2018,2))
cd.grid()
cd.set_title("Server")


"""

Cost change over the years wrt the processor. 

"""
# print(Mobile.loc[Mobile["year"]==2017]["Recommended_Customer_Price"])

MobilePrice=Mobile.groupby('year')["Recommended_Customer_Price"].max().to_frame('MaxPrice')
##print(MobilePrice)
DesktopPrice=Desktop.groupby('year')["Recommended_Customer_Price"].max().to_frame('MaxPrice')
##print(DesktopPrice)
ServerPrice=Server.groupby('year')["Recommended_Customer_Price"].max().to_frame('MaxPrice')
##print(ServerPrice)

h=MobilePrice.plot()
DesktopPrice.plot(ax=h)
ServerPrice.plot(ax=h)
h.set_xticks(np.arange(1999,2018,2))
h.legend(["Mobile", "Desktop","Server"])
h.set_title("Price change")
h.set_ylabel('Price')
h.grid()


"""
Base frequency

"""

# print(Mobile.loc[Mobile["year"]==2010]["Processor_Base_Frequency"])

MobileFreq=Mobile.groupby('year')["Processor_Base_Frequency"].max().to_frame('BaseFreq')
##print(MobileFreq)
DesktopFreq=Desktop.groupby('year')["Processor_Base_Frequency"].max().to_frame('BaseFreq')
##print(DesktopFreq)
ServerFreq=Server.groupby('year')["Processor_Base_Frequency"].max().to_frame('BaseFreq')
##print(ServerFreq)

i=MobileFreq.plot()
DesktopFreq.plot(ax=i)
ServerFreq.plot(ax=i)
i.set_xticks(np.arange(1999,2018,2))
i.legend(["Mobile", "Desktop","Server"])
i.set_title("Base Frequnecy")
i.set_ylabel('Frequency (MHz)')
i.grid()


"""
Change of instructions per second

"""

Mobile["instructions"] = Mobile["Processor_Base_Frequency"] * Mobile["nb_of_Cores"] * Mobile["nb_of_Threads"]/1000
# print(Mobile.loc[Mobile["year"]==2009]["instructions"])
MobileInstruc=Mobile.groupby('year')["instructions"].max().to_frame('InstructionsPerSec')
# #print(MobileInstruc)

Desktop["instructions"] = Desktop["Processor_Base_Frequency"] * Desktop["nb_of_Cores"] * Desktop["nb_of_Threads"]/1000
DesktopInstruc=Desktop.groupby('year')["instructions"].max().to_frame('InstructionsPerSec')
# #print(DesktopInstruc)

Server["instructions"] = Server["Processor_Base_Frequency"] * Server["nb_of_Cores"] * Server["nb_of_Threads"]/1000
ServerInstruc=Server.groupby('year')["instructions"].max().to_frame('InstructionsPerSec')
# #print(ServerInstruc)

j=MobileInstruc.plot()
DesktopInstruc.plot(ax=j)
ServerInstruc.plot(ax=j)
j.set_xticks(np.arange(1999,2018,2))
j.legend(["Mobile", "Desktop","Server"])
j.set_title("Instructions Per Second")
j.set_ylabel('Instructions(Billion)')
j.grid()

"""
TDP vs base frequency

"""

MobileTDP=Mobile.groupby('year')["TDP"].max().to_frame('TDP')
# print(MobileTDP)

idx=Mobile.groupby(['year'])["Processor_Base_Frequency"].transform(max) == Mobile['Processor_Base_Frequency']
MobileFreqTDP = Mobile[idx][["year","TDP"]].set_index('year').sort_index()
# print(MobileFreqTDP)
MobileFreqTDP.rename(columns = {'TDP':'baseFreqTDP'}, inplace = True) 

k = MobileTDP.plot()
MobileFreqTDP.plot(ax=k)
k.set_xticks(np.arange(1999,2018,2))
k.legend(["Max TDP", "TDP with best performance"],loc="upper left")
k.set_title("TDP : Mobile")
k.set_ylabel('TDP (W)')
k.grid()

DesktopTDP=Desktop.groupby('year')["TDP"].max().to_frame('TDP')
# #print(DesktopTDP)

idx=Desktop.groupby(['year'])["Processor_Base_Frequency"].transform(max) == Desktop['Processor_Base_Frequency']
DesktopFreqTDP = Desktop[idx][["year","TDP"]].set_index('year').sort_index()
# #print(DesktopFreqTDP)

k = DesktopTDP.plot()
DesktopFreqTDP.plot(ax=k)
k.set_xticks(np.arange(1999,2018,2))
k.legend(["Max TDP", "TDP with best performance"],loc="upper left")
k.set_title("TDP : Desktop")
k.set_ylabel('TDP (W)')
k.grid()

ServerTDP=Server.groupby('year')["TDP"].max().to_frame('TDP')
# #print(ServerTDP)

idx=Server.groupby(['year'])["Processor_Base_Frequency"].transform(max) == Server['Processor_Base_Frequency']
ServerFreqTDP = Server[idx][["year","TDP"]].set_index('year').sort_index()
# # print(ServerFreqTDP)

k = ServerTDP.plot(grid=True)
ServerFreqTDP.plot(ax=k)
k.set_xticks(np.arange(1999,2018,2))
k.legend(["Max TDP", "TDP with best performance"],loc="upper left")
k.set_title("TDP : Server")
k.set_ylabel('TDP (W)')
k.grid()



print(pd.concat([MobileFreqTDP, MobileTDP], axis=1).corr())

Mobilecost=Desktop.groupby('year')["Recommended_Customer_Price"].max().to_frame('price')
# #print(ServerTDP)

idx=Desktop.groupby(['year'])["Processor_Base_Frequency"].transform(max) == Desktop['Processor_Base_Frequency']
Mobilefreqcost = Desktop[idx][["year","Recommended_Customer_Price"]].set_index('year').sort_index()
Mobilefreqcost=Mobilefreqcost[~Mobilefreqcost.index.duplicated(keep='first')].fillna(method='ffill')
# print(MobilePrice)
print(Mobilecost)
print(Mobilefreqcost)

k = Mobilecost.plot()
Mobilefreqcost.plot(ax=k)
k.set_xticks(np.arange(1999,2018,2))
k.legend(["Max Price", "Price for high base frequency"],loc="upper left")
k.set_title("Price : Desktop")
k.set_ylabel('Price($)')
k.grid()























































































































