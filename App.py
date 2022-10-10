import streamlit as st
import numpy as np
import pandas as pd
import math
import pmdarima as pm
import tensorflow as tf

from pmdarima.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow import keras
from tensorflow.keras import layers

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

def calc_mae(y,y_hat):
    mae = np.abs(y-y_hat)
    return mae.mean()

def calc_mse(y,y_hat):
    mse = (y-y_hat)**2
    return mse.mean()

def calc_rmse(y,y_hat):
    mse = (y-y_hat)**2
    rmse = np.sqrt(mse.mean())
    return rmse

def integrate(apple_orig, apple_diff):
    return (apple_orig.shift()+apple_diff)

def dickey_fuller_test(series):
    result = adfuller(series)
    
    Statistics = result[0]
    p_Value = result[1]
    
    print("Dickey Fuller Test Statistics are:", Statistics)
    print("P - Value is:", p_Value)
    
    if result[1] <= 0.5:
        print("Strong Evidence against the null hypothesis, reject the null hypothesis, Data is stationary")
        return False
    else:
        print("Weak Evidence against the null hypothesis, accepting the null hypothesis, hence data is non-stationary")
        return True

with st.sidebar:
    option = st.selectbox(
    'Select the stock need to be Forecasted',
    ('Apple Inc (AAPL)','Microsoft Corp(MSFT)','Alphabet Inc Class C(GOOG)','Alphabet Inc Class A(GOOGL)','Amazon.Com Inc.(AMZN)','Tesla Inc(TSLA)','Berkshire Hathaway Inc. Class B(BRK.B)Unitedhealth Group Inc(UNH)','Johnson & Johnson(JNJ)','Visa Inc Class A(V)','"Meta Platforms Inc."(META)','Nvidia Corp(NVDA)','Walmart Stores Inc(WMT)','Exxon Mobil Corp(XOM)','Procter & Gamble(PG)','JP Morgan Chase & Co(JPM)','Mastercard Inc Class A(MA)','Eli Lilly(LLY)','Home Depot Inc(HD)','Pfizer Inc(PFE)','Coca-Cola(KO)','Chevron Corp(CVX)','Abbvie Inc(ABBV)','Bank of America Corp(BAC)','Merck & Co Inc(MRK)','Pepsico Inc(PEP)','Costco Wholesale Corp(COST)','Verizon Communications Inc(VZ)','Thermo Fisher Scientific Inc(TMO)','Broadcom Inc.(AVGO)','Mcdonalds Corp(MCD)','Abbott Laboratories(ABT)','Oracle Corp(ORCL)','Danaher Corp(DHR)','Accenture Plc Class A(ACN)','Comcast A Corp(CMCSA)','Cisco Systems Inc(CSCO)','Adobe Inc(ADBE)','T Mobile US Inc(TMUS)','Walt Disney(DIS)','Nike Inc Class B(NKE)','Salesforce.Com Inc(CRM)','Qualcomm Inc(QCOM)','Bristol Myers Squibb(BMY)','Nextera Energy Inc(NEE)','Intel Corporation Corp(INTC)','United Parcel Service Inc Class B(UPS)','Wells Fargo(WFC)','Texas Instrument Inc(TXN)','AT&T Inc.(T)','Philip Morris International Inc(PM)','Linde Plc(LIN)','Raytheon Technologies Corporation(RTX)','Amgen Inc(AMGN)','Morgan Stanley(MS)','Union Pacific Corp(UNP)','Advanced Micro Devices Inc(AMD)','International Business Machines Co(IBM)','CVS Health Corp(CVS)','Medtronic Plc(MDT)','S&P Global Inc(SPGI)','American Tower Corporation(AMT)','Lowes Companies Inc(LOW)','Honeywell International Inc(HON)','Charles Schwab Corporation(SCHW)','"Elevance Health Inc."(ELV)','Lockheed Martin Corp(LMT)','ConocoPhillips(COP)','Intuit Inc(INTU)','American Express(AXP)','Goldman Sachs Group Inc(GS)','Caterpillar Inc(CAT)','Deere & Company(DE)','Starbucks Corp(SBUX)','Blackrock Inc(BLK)','Automatic Data Processing Inc(ADP)','Estee Lauder Inc Class A(EL)','"Prologis Inc"(PLD)','Citigroup Inc(C)','Boeing(BA)','Servicenow Inc(NOW)','Mondelez International Inc(MDLZ)','Cigna Corp(CI)','Duke Energy Corp(DUK)','Paypal Holdings Inc(PYPL)','Zoetis Inc(ZTS)','Applied Material Inc(AMAT)','Analog Devices Inc(ADI)','Charter Communications Inc Class A(CHTR)','Chubb Ltd(CB)','Gilead Sciences Inc(GILD)','Netflix Inc(NFLX)','Southern Co(SO)','Altria Group Inc(MO)','Marsh & McLennan Inc(MMC)','Crown Castle International Corp.(CCI)','Intuitive Surgical Inc(ISRG)','Vertex Pharmaceuticals Inc(VRTX)','3M(MMM)','Stryker Corp(SYK)','CME Group Inc Class A(CME)','Northrop Grumman Corp(NOC)','"TJX Companies Inc."(TJX)','Booking Holdings Inc(BKNG)','Target Corp(TGT)','General Electric(GE)','Becton Dickinson(BDX)','US Bancorp(USB)','Progressive Corp(PGR)','Colgate-Palmolive(CL)','Micron Technology Inc(MU)','Regeneron Pharmaceuticals Inc(REGN)','Dominion Energy Inc(D)','"Moderna Inc."(MRNA)','Sherwin Williams(SHW)','Waste Management Inc(WM)','"PNC Financial Services Group Inc."(PNC)','CSX Corp(CSX)','Edwards Lifesciences Corp(EW)','Truist Financial Corporation(TFC)','Humana Inc(HUM)','Fiserv Inc(FISV)','Activision Blizzard Inc(ATVI)','General Dynamics Corp(GD)','Lam Research Corp(LRCX)','Aon Plc(AON)','Dollar General Corp(DG)','Norfolk Southern Corp(NSC)','Fidelity National Information Services(FIS)','EOG Resources Inc(EOG)','Equinix Inc.(EQIX)','FedEx Corporation(FDX)','Illinois Tool Inc(ITW)','Public Storage(PSA)','Occidental Petroleum Corp(OXY)','Boston Scientific Corp(BSX)','Intercontinental Exchange Inc(ICE)','Monster Beverage Corp(MNST)','Keurig Dr Pepper Inc.(KDP)','Moodys Corp(MCO)','Centene Corp(CNC)','"Eaton Corporation PLC"(ETN)','Pioneer Natural Resource(PXD)','"HCA Healthcare Inc."(HCA)','Air Products And Chemicals Inc(APD)','American Electric Power Inc(AEP)','KLA-Tencor Corporation(KLAC)','Kraft Heinz(KHC)','Constellation Brands Class A(STZ)','Metlife Inc(MET)','Mckesson Corp(MCK)','Synopsys Inc(SNPS)','Sempra Energy(SRE)','Emerson Electric(EMR)','Marriott International Inc(MAR)','General Motors(GM)','Kimberly Clark Corp(KMB)','Hershey Foods(HSY)','Ford Motor Company(F)','General Mills Inc(GIS)','Oreilly Automotive Inc(ORLY)','Schlumberger Nv(SLB)','Sysco Corp(SYY)','Ecolab Inc(ECL)','Marathon Petroleum Corp(MPC)','Newmont Corporation(NEM)','Exelon Corp(EXC)','"L3Harris Technologies Inc."(LHX)','Cadence Design Systems Inc(CDNS)','Autozone Inc(AZO)','NXP Semiconductors Nv(NXPI)','Realty Income Corporation(O)','Valero Energy Corp(VLO)','Roper Technologies Inc(ROP)','Paychex Inc(PAYX)','Capital One Financial Corp(COF)','Republic Services Inc(RSG)','Archer-Daniels-Midland Company(ADM)','"IQVIA Holdings Inc."(IQV)','Cintas Corp(CTAS)','Amphenol Corp(APH)','American International Group Inc(AIG)','Williams Inc(WMB)','Dollar Tree Inc(DLTR)','Phillips 66(PSX)','Xcel Energy Inc(XEL)','Kinder Morgan Inc(KMI)','Travelers Companies Inc(TRV)','Corteva Inc.(CTVA)','Autodesk Inc(ADSK)','Welltower Inc(WELL)','Freeport-McMoRan Inc.(FCX)','TE Connectivity Ltd(TEL)','Dow Inc.(DOW)','Chipotle Mexican Grill Inc(CMG)','Motorola Solutions Inc(MSI)','Agilent Technologies Inc(A)','Digital Realty Trust Inc(DLR)','Aflac Inc(AFL)','SBA Communications Corporation(SBAC)','Electronic Arts Inc(EA)','Arthur J Gallagher(AJG)','Brown Forman Inc Class B(BF.B)','Kroger(KR)','Devon Energy Corp(DVN)','Prudential Financial Inc(PRU)','Microchip Technology Inc(MCHP)','Cognizant Technology Solutions(CTSH)','MSCI Inc(MSCI)','Yum Brands Inc(YUM)','Allstate Corp(ALL)','Warner Bros. Discovery Inc. - Series A(WBD)','Consolidated Edison Inc(ED)','Bank Of New York Mellon Corp(BK)','Resmed Inc(RMD)','HP Inc(HPQ)','Johnson Controls International Plc(JCI)','Baxter International Inc(BAX)','Walgreen Boots Alliance Inc(WBA)','WEC Energy Group Inc(WEC)','Hilton Worldwide Holdings Inc(HLT)','Parker-Hannifin Corp(PH)','Biogen Inc(BIIB)','Global Payments Inc(GPN)','Simon Property Group Inc(SPG)','Idexx Laboratories Inc(IDXX)','Arista Networks Inc(ANET)','Carrier Global Corporation(CARR)','VICI Properties Inc(VICI)','Public Service Enterprise Group Inc(PEG)','Trane Technologies plc(TT)','Tyson Foods Inc Class A(TSN)','Hess Corp(HES)','AmerisourceBergen Corp(ABC)','Transdigm Group Inc(TDG)','Otis Worldwide Corporation(OTIS)','International Flavors & Fragrances(IFF)','Nucor Corp(NUE)','Eversource Energy(ES)','Old Dominion Freight Line Inc(ODFL)','Discover Financial Services(DFS)','Illumina Inc(ILMN)','Verisk Analytics Inc(VRSK)','Twitter Inc(TWTR)','Paccar Inc(PCAR)','Cummins Inc(CMI)','Lyondellbasell Industries NV(LYB)','Baker Hughes Company Class A(BKR)','Fastenal(FAST)','DuPont de Nemours Inc.(DD)','PPG Industries Inc(PPG)','Corning Inc(GLW)','M&T Bank Corp(MTB)','Ross Stores Inc(ROST)','Copart Inc(CPRT)','American Water Works Inc(AWK)','Enphase Energy Inc(ENPH)','"AvalonBay Communities Inc"(AVB)','Equity Residential(EQR)','Hormel Foods Corp(HRL)','Las Vegas Sands Corp(LVS)','First Republic Bank(FRC)','Mettler Toledo Inc(MTD)','Weyerhaeuser Company(WY)','D R Horton Inc(DHI)','T Rowe Price Group Inc(TROW)','Ametek Inc(AME)','Keysight Technologies Inc(KEYS)','Halliburton(HAL)','Kellogg(K)','Ameriprise Finance Inc(AMP)','Oneok Inc(OKE)','CBRE Group Inc(CBRE)','DTE Energy(DTE)','Aptiv Plc(APTV)','Ebay Inc(EBAY)','ON Semiconductor Corp(ON)','Edison International(EIX)','W.W. Grainger Inc(GWW)','Church And Dwight Inc(CHD)','Rockwell Automation Inc(ROK)','Equifax Inc(EFX)','SVB Financial Group(SIVB)','Albemarle Corp(ALB)','Southwest Airlines(LUV)','Tractor Supply(TSCO)','Extra Space Storage Inc(EXR)','Ameren Corp(AEE)','Alexandria Real Estate Equities Inc(ARE)','Entergy Corp(ETR)','Laboratory Corporation Of America(LH)','Lennar Corporation Class A(LEN)','Fifth Third Bancorp(FITB)','McCormick & Co  Non-voting(MKC)','State Street Corp(STT)','West Pharmaceutical Services Inc(WST)','Willis Towers Watson Public Limited Company(WTW)','CDW Corp(CDW)','Duke Realty Corporation(DRE)','Firstenergy Corp(FE)','Ball Corporation(BALL)','Zimmer Biomet Holdings Inc(ZBH)','Coterra Energy Inc.(CTRA)','STERIS PLC(STE)','Ansys Inc(ANSS)','Hartford Financial Services Group(HIG)','Ulta Beauty Inc(ULTA)','PPL Corporation(PPL)','Ventas Inc(VTR)','Waters Corp(WAT)','Northern Trust Corporation(NTRS)','Vulcan Materials(VMC)','Align Technology Inc(ALGN)','Martin Marietta Materials Inc(MLM)','Genuine Parts(GPC)','Fortive Corp(FTV)','Delta Air Lines Inc(DAL)','CMS Energy Corp(CMS)','Mid-America Apartment Communities Inc(MAA)','Verisign Inc(VRSN)','Gartner Inc(IT)','Diamondback Energy Inc(FANG)','Constellation Energy Corporation(CEG)','Garmin Ltd(GRMN)','Live Nation Entertainment Inc(LYV)','Amcor plc(AMCR)','Monolithic Power Systems Inc(MPWR)','Quanta Services Inc(PWR)','Centerpoint Energy Inc(CNP)','Match Group Inc(MTCH)','Raymond James Inc(RJF)','Clorox(CLX)','Catalent Inc(CTLT)','Cincinnati Financial Corp(CINF)','United Rentals Inc(URI)','Incyte Corp(INCY)','Paycom Software Inc(PAYC)','Fox Corporation Class B(FOX)','Fox Corporation Class A(FOXA)Teledyne Technologies Inc(TDY)','VF Corp(VFC)','JB Hunt Transport Services Inc(JBHT)','Citizens Financial Group Inc(CFG)','Hologic Inc(HOLX)','Rollins Inc(ROL)','Broadridge Financial Solutions Inc(BR)','Huntington Bancshares Inc(HBAN)','Essex Property Trust Inc(ESS)','Regions Financial Corp(RF)','Epam Systems Inc(EPAM)','Dover Corp(DOV)','Hewlett Packard Enterprise(HPE)','Perkinelmer Inc(PKI)','CF Industries Holdings Inc(CF)','Molina Healthcare Inc(MOH)','WR Berkley Corp(WRB)','Brown & Brown Inc(BRO)Seagate Technology Plc(STX)','Ingersoll Rand Plc(IR)','Stanley Black & Decker Inc(SWK)','Best Buy Inc(BBY)','Skyworks Solutions Inc(SWKS)','Jacobs Engineering Group Inc.(J)','Expeditors International Of Washington(EXPD)','Conagra Brands Inc(CAG)','Fleetcor Technologies Inc(FLT)','Mosaic(MOS)','Paramount Global - Class B(PARA)','Quest Diagnostics Inc(DGX)','Keycorp(KEY)','International Paper(IP)','Principal Financial Group Inc(PFG)','Atmos Energy Corp(ATO)','Cooper Companies Inc.(COO)','Zebra Technologies Corp(ZBRA)','Factset Research Systems Inc(FDS)','Synchrony Financial(SYF)','SolarEdge Technologies Inc(SEDG)','Evergy Inc(EVRG)','Pool Corp(POOL)','Campbell Soup(CPB)','Cardinal Health Inc(CAH)','Teradyne Inc(TER)','Westinghouse Air Brake Technologies(WAB)','Nasdaq Inc(NDAQ)','Bio Rad Laboratories Inc Class A(BIO)','Alliant Energy Corp(LNT)','Darden Restaurants Inc(DRI)','Marathon Oil Corp(MRO)Western Digital Corp(WDC)','J.M. Smucker(SJM)','Dominos Pizza Inc(DPZ)','NetApp Inc(NTAP)','"NVR Inc."(NVR)','Trimble Inc(TRMB)','Take Two Interactive Software Inc(TTWO)','Carmax Inc(KMX)','LKQ Corp(LKQ)','"UDR Inc"(UDR)','"Healthpeak Properties Inc."(PEAK)','Akamai Technologies Inc(AKAM)','Camden Property Trust(CPT)','Howmet Aerospace Inc.(HWM)','IDEX Corp(IEX)','AES Corp(AES)','Loews Corp(L)','Tyler Technologies Inc(TYL)','Xylem Inc(XYL)','NortonLifeLock Inc.(NLOK)','Expedia Inc(EXPE)','"Jack Henry & Associates Inc."(JKHY)','Leidos Holdings Inc(LDOS)','"Boston Properties Inc"(BXP)','Avery Dennison Corp(AVY)','Bio-Techne Corp(TECH)','Generac Holdings Inc(GNRC)','Omnicom Group Inc(OMC)','Iron Mountain Incorporated(IRM)','Citrix Systems Inc(CTXS)','Textron Inc(TXT)','Packaging Corp Of America(PKG)','Molson Coors Brewing Class B(TAP)','FMC Corp(FMC)','"Cboe Global Markets Inc."(CBOE)','Masco Corp(MAS)','United Continental Holdings Inc(UAL)','"C.H. Robinson Worldwide Inc."(CHRW)','Franklin Resources Inc(BEN)','MGM Resorts International(MGM)','Kimco Realty Corporation(KIM)','Viatris Inc.(VTRS)','Abiomed Inc(ABMD)','PTC Inc(PTC)','Nordson Corp(NDSN)','Nisource Inc(NI)','Celanese Corporation(CE)','Advance Auto Parts Inc(AAP)','"Host Hotels & Resorts Inc"(HST)','Teleflex Inc(TFX)','Eastman Chemical(EMN)','Signature Bank(SBNY)','Hasbro Inc(HAS)','"CenturyLink Inc."(LUMN)','Interpublic Group Of Companies Inc(IPG)','Charles River Laboratories International(CRL)','Lamb Weston Holdings Inc(LW)','Apache Corp(APA)','Snap On Inc(SNA)','Pultegroup Inc(PHM)','Henry Schein Inc(HSIC)','Everest Re Group Ltd(RE)','"Qorvo Inc."(QRVO)','Etsy Inc(ETSY)','Regency Centers Corporation(REG)','Westrock(WRK)','Marketaxess Holdings Inc(MKTX)','Carnival Corp(CCL)','Globe Life Inc.(GL)','Comerica Inc(CMA)','American Airlines Group Inc(AAL)','Fortinet Inc(FTNT)','Assurant Inc(AIZ)','Juniper Networks Inc(JNPR)','Dish Network Corp Class A(DISH)','News Corp Class B(NWS)','News Corp Class A(NWSA)','Whirlpool Corp(WHR)','F5 Networks Inc(FFIV)','A O Smith Corp(AOS)','NRG Energy Inc(NRG)','Universal Health Services Inc.(UHS)','Allegion PLC(ALLE)','Robert Half(RHI)','Nielsen Holdings Plc(NLSN)','Sealed Air Corp(SEE)','Huntington Ingalls Industries Inc(HII)','Fortune Brands Home And Security(FBHS)','Organon & Co.(OGN)','Royal Caribbean Cruises Ltd(RCL)','Pinnacle West Corp(PNW)','Mohawk Industries Inc(MHK)','BorgWarner Inc(BWA)','Lincoln National Corp(LNC)','Tapestry(TPR)','Newell Brands Inc(NWL)','Davita Inc(DVA)','Caesars Entertainment Corp(CZR)','Federal Realty Investment Trust(FRT)','Dexcom Inc(DXCM)','Pentair(PNR)','Dentsply Sirona Inc(XRAY)','Ceridian HCM Holding Inc.(CDAY)','Zions Bancorporation(ZION)','Invesco Ltd(IVZ)','Ralph Lauren Corp Class A(RL)','DXC Technology Company(DXC)','Wynn Resorts Ltd(WYNN)','"Bath & Body Works Inc."(BBWI)','Vornado Realty Trust(VNO)','Alaska Air Group Inc(ALK)','Penn National Gaming Inc(PENN)','PVH Corp(PVH)','Norwegian Cruise Line Holdings Ltd(NCLH)'))
    
    # st.warning(option)


st.sidebar.text("Training Date Range :")
start = st.sidebar.date_input("Training Date Start :")
end = st.sidebar.date_input("Training Date End :")

# st.warning(start)
# st.warning(end)
    

with st.sidebar:
    period = st.slider(
    "Forecast Period",
    
    min_value= 1,
    
    value=164,
    
    max_value= 365,
    )

# st.warning(period)

with st.sidebar:

    ARIMA, LSTM = st.columns(2)
   
    ARIMA_Model = ARIMA.checkbox("ARIMA")  

    LSTM_Model = LSTM.checkbox("LSTM")

    Prophet_M, VAR_M = st.columns(2)

    Prophet_Model = Prophet_M.checkbox("Facebook Prophet")

    VAR_Model = VAR_M.checkbox("Vector AutoRegressor (VAR)")

    Fifth_Model, GluonTs_Model, Sixth_Model = st.columns(3)

    GluonTs_Model.checkbox("Amazon GluonTs")


if VAR_Model:

    st.markdown(body='# Vector Autogressor Model VAR')
    str1 = 'Dataset S&P\\individual_stocks_5yr\\individual_stocks_5yr\\'

    str2 = option.split('(')[1].split(')')[0] + '_data.csv'

    # st.warning(str1+str2)

    df_stock = pd.read_csv(str1+str2)

    stock = df_stock.copy() 

    data_is_non_stationary = True

    differenced_stock_open = stock.open
    differenced_stock_close = stock.close
    differenced_stock_low = stock.low
    differenced_stock_high = stock.high
    differenced_stock_vol = stock.volume

    order_of_differencing = 0
    while(data_is_non_stationary):

        stock_open = dickey_fuller_test(differenced_stock_open)
        stock_close = dickey_fuller_test(differenced_stock_close)
        stock_low = dickey_fuller_test(differenced_stock_low)
        stock_high = dickey_fuller_test(differenced_stock_high)
        stock_vol = dickey_fuller_test(differenced_stock_vol)
        data_is_non_stationary = stock_open or stock_close or stock_high or stock_low or stock_vol

        if data_is_non_stationary :
            order_of_differencing = order_of_differencing + 1
            differenced_stock_open = stock.open.diff().dropna()
            differenced_stock_close = stock.close.diff().dropna()
            differenced_stock_low = stock.high.diff().dropna()
            differenced_stock_high = stock.low.diff().dropna()
            differenced_stock_vol = stock.volume.diff().dropna()
    
    var_stock = pd.concat([differenced_stock_open, differenced_stock_close, differenced_stock_low, differenced_stock_high], axis=1)
        
    stock_train = var_stock[:math.ceil(.85*len(var_stock))]
    stock_test = var_stock[math.ceil(.85*len(var_stock)):]

    var_model_instance = VAR(stock_train)

    order_for_var = var_model_instance.select_order(40).bic

    var_result = var_model_instance.fit(order_for_var)

    var_result.summary()

    lag = var_result.k_ar

    result_forecast = var_result.forecast(stock_test.values[-lag:], steps=len(stock_test))

    var_output = pd.concat([pd.DataFrame(stock_train.close.values), pd.DataFrame((result_forecast[:,0]))])

    actual_output = integrate(pd.DataFrame(stock.close.values),var_output).dropna()

    actual_output = actual_output.reset_index(drop=True)
    actual_pred = actual_output.reset_index(drop=True)[-len(stock_test):]

    st.write('VAR Actual Dataframe')
    st.write(pd.DataFrame(stock.close))
    st.write(type(pd.DataFrame(stock.close)))
    st.write(len(pd.DataFrame(stock.close)))

    st.write('VAR Predicted Dataframe')
    st.write(actual_pred)
    st.write(type(actual_pred))
    st.write(len(actual_pred))
    
    var_df = pd.DataFrame(stock.close)
    st.write(type(var_df))
    st.write(type(actual_pred))
    var_df['yhat'] = actual_pred

    st.line_chart(var_df)
    
    mae_test, mae = st.columns(2)
    mae_test.text('Mean Absolute Error is:')
    mae.write(calc_mae(stock.close[-len(actual_pred):].values, actual_pred.values))

    mse_test, mse = st.columns(2)
    mse_test.text('Mean Square Error is:')
    mse.write(calc_mse(stock.close[-len(actual_pred):].values, actual_pred.values))

    rmse_test, rmse = st.columns(2)
    rmse_test.text('Root Mean Absolute Error is:')
    rmse.write(calc_rmse(stock.close[-len(actual_pred):].values, actual_pred.values))
    
if LSTM_Model :
        st.markdown(body="# Long Short Term Memory Model:")

        str1 = 'Dataset S&P\\individual_stocks_5yr\\individual_stocks_5yr\\'

        str2 = option.split('(')[1].split(')')[0] + '_data.csv'

        # st.warning(str1+str2)

        df_stock = pd.read_csv(str1+str2)

        stock = df_stock.copy()

        stock_lstm = stock[['close']]

        features = stock[['close']] 
        target = stock['close'].tolist() 

        x_train, x_test, y_train, y_test = train_test_split(features,target, test_size=165)

        window_length = 1
        feature_count = 1

        train_generator = TimeseriesGenerator(x_train, y_train, length=window_length, sampling_rate=1, batch_size=1, stride = 1)
        test_generator = TimeseriesGenerator(x_test, y_test, length=window_length, sampling_rate=1, batch_size=1, stride = 1)

        LSTM_Model_Instance = keras.Sequential([
            keras.Input(shape=(window_length, feature_count)),
            layers.LSTM(128, activation = 'relu', return_sequences = True),
            layers.LSTM(128, activation = 'relu', return_sequences = True),
            layers.LSTM(64, activation = 'relu', return_sequences = False),
            layers.Dense(1)])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience = 2,
                                                         mode=min)
        LSTM_Model_Instance.compile(loss=tf.losses.MeanSquaredError(),
             optimizer = tf.optimizers.Adam(),
             metrics = [tf.metrics.MeanAbsoluteError()])
       
        history = LSTM_Model_Instance.fit_generator(train_generator, epochs=50,
                             validation_data=test_generator,
                             shuffle=False, callbacks=[early_stopping])
        
        LSTM_Model_Instance.evaluate_generator(test_generator, verbose=0)

        LSTM_prediction = LSTM_Model_Instance.predict_generator(test_generator)

        LSTM_prediction_df = pd.DataFrame(LSTM_prediction)

        rangeIndex = pd.RangeIndex(start=len(x_train), stop=len(stock)-1, step=1)
        LSTM_prediction_df = LSTM_prediction_df.set_index(rangeIndex)
        lstm_df = pd.DataFrame(stock_lstm)
        lstm_df['yhat'] = LSTM_prediction_df
        st.line_chart(lstm_df, use_container_width=True)

        mae_test, mae = st.columns(2)
        mae_test.text('Mean Absolute Error is:')
        mae.write(calc_mae(stock.close[-len(x_train):].values, LSTM_prediction))

        mse_test, mse = st.columns(2)
        mse_test.text('Mean Square Error is:')
        mse.write(calc_mse(stock.close[-len(x_train):].values, LSTM_prediction))

        rmse_test, rmse = st.columns(2)
        rmse_test.text('Root Mean Absolute Error is:')
        rmse.write(calc_rmse(stock.close[-len(x_train):].values, LSTM_prediction))

if ARIMA_Model :
    st.markdown(body="## AutoRegression Integrated Moving Average Model :")

    str1 = 'Dataset S&P\\individual_stocks_5yr\\individual_stocks_5yr\\'

    str2 = option.split('(')[1].split(')')[0] + '_data.csv'

    # st.warning(str1+str2)

    df_stock = pd.read_csv(str1+str2)

    stock = df_stock.copy()

    stock_train = stock[:math.ceil(.85*len(stock))]
    stock_test = stock[math.ceil(.85*len(stock)):]

    Arima_Model_Instance = pm.auto_arima(stock['close'][:len(stock_train)], start_p = 0,  start_q=0, max_order=4, test='adf',
                            error_action='ignore', suppress_warning = True, stepwise = True, trace = True)

    prediction_Arima = Arima_Model_Instance.predict(len(stock_test), return_conf_int=True)

    ARIMA_prediction_df = pd.DataFrame(prediction_Arima[0])

    rangeIndex = pd.RangeIndex(start=len(stock_train), stop=len(stock), step=1)
    ARIMA_prediction_df = ARIMA_prediction_df.set_index(rangeIndex)
    
    arima_df = pd.DataFrame(stock.close)
    arima_df['yhat'] = ARIMA_prediction_df
    st.line_chart(arima_df, use_container_width=True)

    st.write(len(arima_df))
    st.write(len(ARIMA_prediction_df))


    mae_test, mae = st.columns(2)
    mae_test.text('Mean Absolute Error is:')
    mae.write(calc_mae(stock.close[-len(arima_df):].values, ARIMA_prediction_df.values))

    mse_test, mse = st.columns(2)
    mse_test.text('Mean Square Error is:')
    mse.write(calc_mse(stock.close[-len(arima_df):].values, ARIMA_prediction_df.values))

    rmse_test, rmse = st.columns(2)
    rmse_test.text('Root Mean Absolute Error is:')
    rmse.write(calc_rmse(stock.close[-len(arima_df):].values, ARIMA_prediction_df.values    ))



