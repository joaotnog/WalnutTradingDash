




def get_ret(x, data):
    # x = dd_details.iloc[0]
    st1 = data[data.index==x.Start].Close[0]
    st2 = data[data.index==x.End].Close[0]
    ret = round(st2/st1, 3)-1
    return ret

def get_median_ret(data, t):
    data_ = data.copy()
    data_ = data_[::t]
    median_ret = (data_.Close/data_.shift(1).Close-1).median().round(3)
    return median_ret

def proc_dd_details(dd_details, data):
    dd_details_ = dd_details.copy()
    dd_details_ = dd_details_[['Start','End','Length']]
    dd_details_['Return'] = dd_details_.apply(lambda x:get_ret(x,data), axis=1)
    dd_details_['Typical Buy&Hold return (same duration)'] = dd_details_.apply(lambda x:get_median_ret(data, x.Length), axis=1)
    dd_details_['Timing quality'] = dd_details_['Typical Buy&Hold return (same duration)']<dd_details_['Return']
    dd_details_['Timing quality'] = ['Good' if x==True else 'Bad' for x in dd_details_['Timing quality']]
    dd_details_.columns=['Buy date', 'Sell date', 'Duration', 'Return',
           'Typical Buy&Hold return (same duration)', 'Timing quality']
    return dd_details_