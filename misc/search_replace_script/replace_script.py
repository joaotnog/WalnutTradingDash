from artifacts.mapping import *

# Read in the file
with open('misc/search_replace_script/Functions_before.py', 'r') as file :
    filedata = file.read()

for key, value in map_shorttech2tech.items():
    filedata = filedata.replace("'{}'".format(key), "'{}'".format(value))
    
replace_dict = dict()
replace_dict["offset = int(offset.text_"]="offset=0 #offset = int(offset.text_"
replace_dict["'<, Crossing Down'"]="'LOWER THAN'"
replace_dict["'>, Crossing Up'"]="'HIGHER THAN'"
replace_dict["'==, Equal To'"]="'EQUAL TO'"
replace_dict["exit_input_1 = exit_condition_inputs.selectbox('Input 1'"]="exit_input_1 = exit_condition_inputs.selectbox('AND SELL WHEN'"
replace_dict["entry_input_1 = entry_condition_inputs.selectbox('Input 1'"]="entry_input_1 = entry_condition_inputs.selectbox('BUY WHEN'"
replace_dict["'Comparator'"]="'IS'"
replace_dict["'Input 2'"]="'INDICATOR'"
replace_dict["num_stream.sidebar.markdown('')"]="#num_stream.sidebar.markdown('')"
replace_dict["exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)"]="exit_condition_inputs=entry_condition_inputs #exit_condition_inputs = num_stream.sidebar.expander('EXIT CONDITION', False)"
replace_dict["ENTRY CONDITION"]="TRADING STRATEGY"
replace_dict["'Specify Input Value'"]="''"
replace_dict["entry_input_2 = entry_condition_inputs.selectbox('INDICATOR', inputs2"]="entry_input_2 = inputs2[0] #entry_input_2 = entry_condition_inputs.selectbox('INDICATOR', inputs2"
replace_dict["exit_input_2 = exit_condition_inputs.selectbox('INDICATOR', inputs2"]="exit_input_2 = inputs2[0] #exit_input_2 = exit_condition_inputs.selectbox('INDICATOR', inputs2"
replace_dict["matype = matype.selectbox('MA Type', mas"]="matype='sma' #matype = matype.selectbox('MA Type', mas"
replace_dict["entry_condition_inputs = num_stream.sidebar.expander('TRADING STRATEGY', False)"]="entry_condition_inputs = num_stream.sidebar; entry_condition_inputs.header('With the strategy')"
replace_dict["entry_input_1 = entry_condition_inputs.selectbox('BUY WHEN', inputs1"]="entry_condition_inputs.text('BUY when the '); entry_input_1 = inputs1[0] #entry_input_1 = entry_condition_inputs.selectbox('BUY WHEN', inputs1"
replace_dict["exit_input_1 = exit_condition_inputs.selectbox('AND SELL WHEN', inputs1"]="exit_condition_inputs.text('and SELL when the '); exit_input_1 = inputs1[0] #exit_input_1 = exit_condition_inputs.selectbox('AND SELL WHEN', inputs1"
replace_dict["exit_comparator = exit_condition_inputs.selectbox('IS', exit_conditions"]="exit_comparator = exit_condition_inputs.selectbox('', exit_conditions"
replace_dict["entry_comparator = entry_condition_inputs.selectbox('IS', entry_conditions"]="entry_comparator = entry_condition_inputs.selectbox('', entry_conditions"
replace_dict["HIGHER THAN"]="is higher than"
replace_dict["LOWER THAN"]="is lower than"
replace_dict["EQUAL TO"]="is equal to"




replace_dict["d_period = int(d_period.text_input('D Period', value = 3, "]="d_period=3 #d_period = int(d_period.text_input('D Period', value = 3, "
replace_dict["k_period = int(k_period.text_input('K Period', value = 3, "]="k_period=3 #k_period = int(k_period.text_input('K Period', value = 3, "
replace_dict["period = int(period.text_input('-VI Period', value = 14, k"]="period=14 #period = int(period.text_input('-VI Period', value = 14, k"
replace_dict["period = int(period.text_input('CI Period', value = 14, ke"]="period=14 #period = int(period.text_input('CI Period', value = 14, ke"
replace_dict["period = int(period.text_input('CI Period', value = 50, ke"]="period=50 #period = int(period.text_input('CI Period', value = 50, ke"
replace_dict["period = int(period.text_input('Period', value = 20, key ="]="period=20 #period = int(period.text_input('Period', value = 20, key ="#
replace_dict["period = int(period.text_input('Period', value = 30, key ="]="period=30 #period = int(period.text_input('Period', value = 30, key ="#
replace_dict["period = int(period.text_input('W%R Period', value = 14, k"]="period=14 #period = int(period.text_input('W%R Period', value = 14, k"
replace_dict["period = int(period.text_input('cfo Period', value = 25, k"]="period=25 #period = int(period.text_input('cfo Period', value = 25, k"
replace_dict["period = int(period.text_input('ci Period', value = 50, ke"]="period=50 #period = int(period.text_input('ci Period', value = 50, ke"
replace_dict["period = int(period.text_input('tema Period', value = 20, "]="period=20 #period = int(period.text_input('tema Period', value = 20, "
replace_dict["signal_period = int(signal_period.text_input('Signal Perio"]="signal_period=9 #signal_period = int(signal_period.text_input('Signal Perio"

def replace_helper(filedata):
    z = list(set([x[:100].split(', key')[0].lstrip() for x in filedata.split('\n') if ".text_input('" in x and not 'period' in x]))                                 
    zvals = [zi.split('value = ')[1] for zi in z]
    zkeys = [zi.split(" = ")[0] for zi in z]
    return ['replace_dict["'+zi+'"]="'+zkeysi+'='+zvalsi+' #'+zi+'"' for zi, zvalsi, zkeysi in zip(z,zvals,zkeys)]


replace_dict["std = int(std.text_input('Standard Deviation', value = 2"]="std=2 #std = int(std.text_input('Standard Deviation', value = 2"
replace_dict["signal = int(signal.text_input('Signal', value = 9"]="signal=9 #signal = int(signal.text_input('Signal', value = 9"
replace_dict["slow_ma = int(slow_ma.text_input('Slow MA', value = 26"]="slow_ma=26 #slow_ma = int(slow_ma.text_input('Slow MA', value = 26"
replace_dict["fast = int(fast.text_input('Fast', value = 7"]="fast=7 #fast = int(fast.text_input('Fast', value = 7"
replace_dict["max_af = float(max_af.text_input('Max AF', value = 0.2"]="max_af=0.2 #max_af = float(max_af.text_input('Max AF', value = 0.2"
replace_dict["min_af = float(min_af.text_input('Min AF', value = 0.02"]="min_af=0.02 #min_af = float(min_af.text_input('Min AF', value = 0.02"
replace_dict["max_af = float(max_af.text_input('Max AF', value = '0.2'"]="max_af='0.2' #max_af = float(max_af.text_input('Max AF', value = '0.2'"
replace_dict["fast_ma = int(fast_ma.text_input('Fast MA', value = 12"]="fast_ma=12 #fast_ma = int(fast_ma.text_input('Fast MA', value = 12"
replace_dict["medium = int(medium.text_input('Medium', value = 14"]="medium=14 #medium = int(medium.text_input('Medium', value = 14"
replace_dict["min_af = float(min_af.text_input('Min AF', value = '0.02'"]="min_af='0.02' #min_af = float(min_af.text_input('Min AF', value = '0.02'"
replace_dict["slow = int(offset.text_input('Long ROC', value = 14"]="slow=14 #slow = int(offset.text_input('Long ROC', value = 14"
replace_dict["scalar = int(scalar.text_input('Scalar (Optional)', value = 100"]="scalar=100 #scalar = int(scalar.text_input('Scalar (Optional)', value = 100"
replace_dict["slow = int(slow.text_input('Slow', value = 28"]="slow=28 #slow = int(slow.text_input('Slow', value = 28"
replace_dict["multiplier = int(multiplier.text_input('SuperTrend Multiplier', value = 3"]="multiplier=3 #multiplier = int(multiplier.text_input('SuperTrend Multiplier', value = 3"



for shortech in map_shortech2fun.keys():
    replace_dict["period.text_input('{} Period'".format(shortech)]="period.slider('x days indicator'"


for key, value in replace_dict.items():
    filedata = filedata.replace(key, value)

# re.sub("'.* Period'","'FOR X DAYS'","" + filedata + "")
# re.sub(".text_input\('SuperTrend Period'",".text_input('FOR X DAYS'",filedata)

# Write the file out again
with open('Functions.py', 'w') as file:
    file.write(filedata)