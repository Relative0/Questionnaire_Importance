
import sys

# Choose which bin array will be used.
def DimensionalBinChoice(List,BinList):
    BinVal = []
    if len(BinList) == 1:
        BinVal = DimensionalityBinning_2(List,BinList)
    elif len(BinList) == 2:
        BinVal = DimensionalityBinning_3(List, BinList)
    elif len(BinList) == 3:
        BinVal = DimensionalityBinning_4(List, BinList)
    elif len(BinList) == 4:
        BinVal = DimensionalityBinning_5(List, BinList)
    elif len(BinList) == 5:
        BinVal = DimensionalityBinning_6(List, BinList)
    elif len(BinList) == 6:
        BinVal = DimensionalityBinning_7(List, BinList)
    elif len(BinList) == 7:
        BinVal = DimensionalityBinning_8(List, BinList)
    else:
        print("Issue in Dimensional Binning Choice")
    return BinVal

# A bin array of 8 partitions.
def DimensionalityBinning_8(List, BinList):
    Categories = []
    Mean = List.mean()
    StDev = List.std()
    for value in List:
        if value > (Mean + BinList[6] * StDev):  
            Categories.append(8)
        elif ((Mean + (BinList[6] * StDev)) >= value) and (value > (Mean + BinList[5] * StDev)): 
            Categories.append(7)
        elif ((Mean + BinList[5] * StDev) >= value) and (value > (Mean + BinList[4] * StDev)):  
            Categories.append(6)
        elif ((Mean + BinList[4] * StDev) >= value) and (value > (Mean + BinList[3] * StDev)):  
            Categories.append(5)
        elif ((Mean + BinList[3] * StDev) >= value) and (value > (Mean + BinList[2] * StDev)):  
            Categories.append(4)
        elif ((Mean + BinList[2] * StDev) >= value) and (value > (Mean + BinList[1] * StDev)):  
            Categories.append(3)
        elif ((Mean + BinList[1] * StDev) >= value) and (value > (Mean + BinList[0] * StDev)):  
            Categories.append(2)
        elif ((Mean + BinList[0] * StDev) >= value):  
            Categories.append(1)
        else:
            sys.exit("Error in assigning Category")
    return Categories

# A bin array of 7 partitions.
def DimensionalityBinning_7(List, BinList):
    Categories = []
    Mean = List.mean()
    StDev = List.std()
    for value in List:
        if value > (Mean + BinList[5] * StDev):  
            Categories.append(7)
        elif ((Mean + BinList[5] * StDev) >= value) and (value > (Mean + BinList[4] * StDev)):  
            Categories.append(6)
        elif ((Mean + BinList[4] * StDev) >= value) and (value > (Mean + BinList[3] * StDev)):  
            Categories.append(5)
        elif ((Mean + BinList[3] * StDev) >= value) and (value > (Mean + BinList[2] * StDev)):  
            Categories.append(4)
        elif ((Mean + BinList[2] * StDev) >= value) and (value > (Mean + BinList[1] * StDev)):  
            Categories.append(3)
        elif ((Mean + BinList[1] * StDev) >= value) and (value > (Mean + BinList[0] * StDev)):  
            Categories.append(2)
        elif ((Mean + BinList[0] * StDev) >= value):  
            Categories.append(1)
        else:
            sys.exit("Error in assigning Category")
    return Categories

# A bin array of 6 partitions.
def DimensionalityBinning_6(List, BinList):
    Categories = []
    Mean = List.mean()
    StDev = List.std()
    for value in List:
        if value > (Mean + BinList[4] * StDev):  
            Categories.append(6)
        elif ((Mean + BinList[4] * StDev) >= value) and (value > (Mean + BinList[3] * StDev)):  
            Categories.append(5)
        elif ((Mean + BinList[3] * StDev) >= value) and (value > (Mean + BinList[2] * StDev)):  
            Categories.append(4)
        elif ((Mean + BinList[2] * StDev) >= value) and (value > (Mean + BinList[1] * StDev)):  
            Categories.append(3)
        elif ((Mean + BinList[1] * StDev) >= value) and (value > (Mean + BinList[0] * StDev)):  
            Categories.append(2)
        elif ((Mean + BinList[0] * StDev) >= value):  
            Categories.append(1)
        else:
            sys.exit("Error in assigning Category")
    return Categories

# A bin array of 5 partitions.
def DimensionalityBinning_5(List, BinList):
    Categories = []
    Mean = List.mean()
    StDev = List.std()
    for value in List:
        if value > (Mean + BinList[3] * StDev):  
            Categories.append(5)
        elif ((Mean + BinList[3] * StDev) >= value) and (value > (Mean + BinList[2] * StDev)):  
            Categories.append(4)
        elif ((Mean + BinList[2] * StDev) >= value) and (value > (Mean + BinList[1] * StDev)):  
            Categories.append(3)
        elif ((Mean + BinList[1] * StDev) >= value) and (value > (Mean + BinList[0] * StDev)):  
            Categories.append(2)
        elif ((Mean + BinList[0] * StDev) >= value):  
            Categories.append(1)
        else:
            sys.exit("Error in assigning Category")
    return Categories

# A bin array of 4 partitions.
def DimensionalityBinning_4(List, BinList):
    Categories = []
    Mean = List.mean()
    StDev = List.std()
    for value in List:
        if value > (Mean + BinList[2] * StDev):  
            Categories.append(4)
        elif ((Mean + BinList[2] * StDev) >= value) and (value > (Mean + BinList[1] * StDev)):  
            Categories.append(3)
        elif ((Mean + BinList[1] * StDev) >= value) and (value > (Mean + BinList[0] * StDev)):  
            Categories.append(2)
        elif ((Mean + BinList[0] * StDev) >= value):  
            Categories.append(1)
        else:
            sys.exit("Error in assigning Category")
    return Categories

# A bin array of 3 partitions.
def DimensionalityBinning_3(List, BinList):
    Categories = []
    Mean = List.mean()
    StDev = List.std()
    for value in List:
        if value > (Mean + BinList[1] * StDev):  
            Categories.append(3)
        elif ((Mean + BinList[1] * StDev) >= value) and (value > (Mean + BinList[0] * StDev)):  
            Categories.append(2)
        elif ((Mean + BinList[0] * StDev) >= value):  
            Categories.append(1)
        else:
            sys.exit("Error in assigning Category")
    return Categories

# A bin array of 2 partitions.
def DimensionalityBinning_2(List, BinList):
    Categories = []
    Mean = List.mean()
    StDev = List.std()
    for value in List:
        if value > (Mean + BinList[0] * StDev):  
            Categories.append(2)
        elif ((Mean + BinList[0] * StDev) >= value):  
            Categories.append(1)
        else:
            sys.exit("Error in assigning Category")
    return Categories

