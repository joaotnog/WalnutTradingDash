map_code2crypto = dict(
BTC='Bitcoin',
ETH='Ethereum',
BUSD='BUSD',
USDT='Tether',
BNB='Binance Coin',
XRP='XRP',
SOL='Solana',
AVAX='Avalanche',
LINK='Chainlink',
ADA='Cardano',
USDC='USD Coin',
FIL='FileCoin',
DOT='Polkadot',
SHIB='Shiba Inu',
DOGE='Dogecoin',
MATIC='Polygon',
OP='Optimism',
NEAR='Near',
DAI='Dai',
ATOM='Cosmos',
TUSD='True USD',
WAVES='Waves',
FLOW='Flow - Dapper Labs',
SAND='The Sandbox',
ZIL='Zilliqa',
TRX='TRON',
LTC='Litecoin',
FTM='Fantom',
MANA='Decentraland',
SPELL='Spell Token',
ZEC='ZCash',
RUNE='Thorchain',
ROSE='Oasis Labs',
EOS='EOS',
APE='ApeCoin',
XMR='Monero',
UNI='Uniswap Protocol Token',
XLM='Stellar',
ALGO='Algorand',
AXS='Axie Infinity Shards',
REN='REN',
GMT='STEPN',
BCH='Bitcoin Cash',
RSR='Reserve Rights',
ENS='Ethereum Name Service',
THETA='Theta',
CHZ='Chiliz',
APM='apM Coin',
MINA='Mina Protocol',
CRV='Curve DAO Token',
DGB='DigiByte',
BAT='Basic Attention Token',
FTT='FTX Token',
TRB='Tellor',
GRT='The Graph',
JASMY='JasmyCoin',
VET='VeChain',
KAVA='Kava',
AAVE='Aave',
IMX='Immutable X',
LRC='Loopring',
ONE='Harmony',
TORN='Tornado Cash',
ENJ='Enjin Coin',
SRM='Serum',
LUNC='Terra Classic',
WBTC='Wrapped Bitcoin',
QTUM='QTUM',
EGLD='Elrond',
OCEAN='Ocean Protocol',
PLA='PlayDapp',
DASH='Dash',
DYDX='dYdX',
KSM='Kusama',
ICP='Internet Computer',
WIN='WINk',
UNFI='Unifi Protocol DAO',
YFI='yearn.finance',
JST='JUST',
HBAR='Hedera Hashgraph',
SNX='Synthetix',
NFT='APENFT',
CAKE='PancakeSwap',
SUSHI='Sushi',
SXP='Swipe',
QNT='Quant',
AUCTION='Bounce Finance Governance Token',
CEL='Celsius Network',
RVN='Ravencoin',
WOM='WOM',
KDA='Kadena',
OMG='OMG Network',
LDO='Lido DAO',
KNC='Kyber Network Crystal v2',
LUNA='Terra',
HIVE='Hive',
ANC='Anchor Protocol',
NEO='NEO',
STORJ='Storj',
)

map_crypto2code = {value:key for key, value in map_code2crypto.items()}


map_func2tech = dict(
implement_simple_moving_average= 'Simple Moving Average (SMA)',    
implement_negative_directional_index= '-DI, Negative Directional Index',
implement_normalized_average_true_range= 'Normalized Average True Range (NATR)',
implement_average_directional_index= 'Average Directional Index (ADX)',
implement_stochastic_oscillator_fast= 'Stochastic Oscillator Fast (SOF)',
implement_stochastic_oscillator_slow= 'Stochastic Oscillator Slow (SOS)',
implement_weighted_moving_average= 'Weighted Moving Average (WMA)',
implement_momentum_indicator= 'Momentum Indicator (MOM)',
implement_vortex_indicator= 'Vortex Indicator (VI)',
implement_chande_momentum_oscillator= 'Chande Momentum Oscillator (CMO)',
implement_exponential_moving_average= 'Exponential Moving Average (EMA)',
implement_triple_exponential_moving_average= 'Triple Exponential Moving Average (TEMA)',
implement_double_exponential_moving_average= 'Double Exponential Moving Average (DEMA)',
implement_supertrend= 'SuperTrend',
implement_triangular_moving_average= 'Triangular Moving Average (TRIMA)',
implement_chande_forecast_oscillator= 'Chande Forecast Oscillator (CFO)',
implement_choppiness_index= 'Choppiness Index',
implement_aroon_down= 'Aroon Down',
implement_average_true_range= 'Average True Range (ATR)',
implement_williamsr= 'Williams %R',
implement_parabolic_sar= 'Parabolic SAR',
implement_coppock_curve= 'Coppock Curve',
implement_positive_directional_index= '+DI, Positive Directional Index',
implement_rsi= 'Relative Strength Index (RSI)',
implement_macd_signal= 'MACD Signal',
implement_aroon_oscillator= 'Aroon Oscillator',
implement_stochrsi_fastk= 'Stochastic RSI FastK',
implement_stochrsi_fastd= 'Stochastic RSI FastD',
implement_ultimate_oscillator= 'Ultimate Oscillator',
implement_aroon_up= 'Aroon Up',
implement_bollinger_bands= 'Bollinger Bands',
implement_trix= 'TRIX',
implement_cci= 'Commodity Channel Index (CCI)',
implement_macd= 'MACD',
implement_macd_histogram= 'MACD Histogram',
implement_mfi= 'Money Flow Index (MFI)'    
)


map_tech2fun = {value:key for key, value in map_func2tech.items()}