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
for shortech in map_shortech2fun.keys():
    replace_dict["period.text_input('{} Period'".format(shortech)]="period.text_input('(X-DAYS)'"


for key, value in replace_dict.items():
    filedata = filedata.replace(key, value)

# re.sub("'.* Period'","'FOR X DAYS'","" + filedata + "")
# re.sub(".text_input\('SuperTrend Period'",".text_input('FOR X DAYS'",filedata)

# Write the file out again
with open('Functions.py', 'w') as file:
    file.write(filedata)