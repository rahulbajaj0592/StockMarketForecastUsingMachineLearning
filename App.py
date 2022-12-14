from operator import concat
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

from prophet import Prophet

from datetime import date, datetime
from datetime import timedelta

from statsmodels.tsa.statespace.varmax import VARMAX

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

def display_metrices(mae, mse, rmse):
    data = [['Mean Square', mae], ['Mean Square', mse], ['Root Mean Square Error', rmse]]
  
    df_metrices = pd.DataFrame(data, columns=['Metrices', 'Value'])

    st.dataframe(df_metrices)
  
    

with st.sidebar:
    option = st.selectbox(
    'Select the stock need to be Forecasted',
    ('Apple Inc (AAPL)','Microsoft Corp(MSFT)','Alphabet Inc Class C(GOOG)','Alphabet Inc Class A(GOOGL)','Amazon.Com Inc.(AMZN)','Berkshire Hathaway Inc. Class B(BRK.B)','Unitedhealth Group Inc(UNH)','Johnson & Johnson(JNJ)','Visa Inc Class A(V)','Meta Platforms Inc.(FB)','Nvidia Corp(NVDA)','Walmart Stores Inc(WMT)','Exxon Mobil Corp(XOM)','Procter & Gamble(PG)','JP Morgan Chase & Co(JPM)','Mastercard Inc Class A(MA)','Eli Lilly(LLY)','Home Depot Inc(HD)','Pfizer Inc(PFE)','Coca-Cola(KO)','Chevron Corp(CVX)','Abbvie Inc(ABBV)','Bank of America Corp(BAC)','Merck & Co Inc(MRK)','Pepsico Inc(PEP)','Costco Wholesale Corp(COST)','Verizon Communications Inc(VZ)','Thermo Fisher Scientific Inc(TMO)','Broadcom Inc.(AVGO)','Mcdonalds Corp(MCD)','Abbott Laboratories(ABT)','Oracle Corp(ORCL)','Danaher Corp(DHR)','Accenture Plc Class A(ACN)','Comcast A Corp(CMCSA)','Cisco Systems Inc(CSCO)','Adobe Inc(ADBE)','Walt Disney(DIS)','Nike Inc Class B(NKE)','Salesforce.Com Inc(CRM)','Qualcomm Inc(QCOM)','Bristol Myers Squibb(BMY)','Nextera Energy Inc(NEE)','Intel Corporation Corp(INTC)','United Parcel Service Inc Class B(UPS)','Wells Fargo(WFC)','Texas Instrument Inc(TXN)','AT&T Inc.(T)','Philip Morris International Inc(PM)','Amgen Inc(AMGN)','Morgan Stanley(MS)','Union Pacific Corp(UNP)','Advanced Micro Devices Inc(AMD)','International Business Machines Co(IBM)','CVS Health Corp(CVS)','Medtronic Plc(MDT)','S&P Global Inc(SPGI)','American Tower Corporation(AMT)','Lowes Companies Inc(LOW)','Honeywell International Inc(HON)','Charles Schwab Corporation(SCHW)','Lockheed Martin Corp(LMT)','ConocoPhillips(COP)','Intuit Inc(INTU)','American Express(AXP)','Goldman Sachs Group Inc(GS)','Caterpillar Inc(CAT)','Deere & Company(DE)','Starbucks Corp(SBUX)','Blackrock Inc(BLK)','Automatic Data Processing Inc(ADP)','Estee Lauder Inc Class A(EL)','Prologis Inc(PLD)','Citigroup Inc(C)','Boeing(BA)','Mondelez International Inc(MDLZ)','Cigna Corp(CI)','Duke Energy Corp(DUK)','Paypal Holdings Inc(PYPL)','Zoetis Inc(ZTS)','Applied Material Inc(AMAT)','Analog Devices Inc(ADI)','Charter Communications Inc Class A(CHTR)','Chubb Ltd(CB)','Gilead Sciences Inc(GILD)','Netflix Inc(NFLX)','Southern Co(SO)','Altria Group Inc(MO)','Marsh & McLennan Inc(MMC)','Crown Castle International Corp.(CCI)','Intuitive Surgical Inc(ISRG)','Vertex Pharmaceuticals Inc(VRTX)','3M(MMM)','Stryker Corp(SYK)','CME Group Inc Class A(CME)','Northrop Grumman Corp(NOC)','TJX Companies Inc.(TJX)','Target Corp(TGT)','General Electric(GE)','Becton Dickinson(BDX)','US Bancorp(USB)','Progressive Corp(PGR)','Colgate-Palmolive(CL)','Micron Technology Inc(MU)','Regeneron Pharmaceuticals Inc(REGN)','Dominion Energy Inc(D)','Sherwin Williams(SHW)','Waste Management Inc(WM)','PNC Financial Services Group Inc.(PNC)','CSX Corp(CSX)','Edwards Lifesciences Corp(EW)','Humana Inc(HUM)','Fiserv Inc(FISV)','Activision Blizzard Inc(ATVI)','General Dynamics Corp(GD)','Lam Research Corp(LRCX)','Aon Plc(AON)','Dollar General Corp(DG)','Norfolk Southern Corp(NSC)','Fidelity National Information Services(FIS)','EOG Resources Inc(EOG)','Equinix Inc.(EQIX)','FedEx Corporation(FDX)','Illinois Tool Inc(ITW)','Public Storage(PSA)','Occidental Petroleum Corp(OXY)','Boston Scientific Corp(BSX)','Intercontinental Exchange Inc(ICE)','Monster Beverage Corp(MNST)','Moodys Corp(MCO)','Centene Corp(CNC)','Eaton Corporation PLC(ETN)','Pioneer Natural Resource(PXD)','HCA Healthcare Inc.(HCA)','Air Products And Chemicals Inc(APD)','American Electric Power Inc(AEP)','KLA-Tencor Corporation(KLAC)','Kraft Heinz(KHC)','Constellation Brands Class A(STZ)','Metlife Inc(MET)','Mckesson Corp(MCK)','Synopsys Inc(SNPS)','Sempra Energy(SRE)','Emerson Electric(EMR)','Marriott International Inc(MAR)','General Motors(GM)','Kimberly Clark Corp(KMB)','Hershey Foods(HSY)','Ford Motor Company(F)','General Mills Inc(GIS)','Oreilly Automotive Inc(ORLY)','Schlumberger Nv(SLB)','Sysco Corp(SYY)','Ecolab Inc(ECL)','Marathon Petroleum Corp(MPC)','Newmont Corporation(NEM)','Exelon Corp(EXC)','Cadence Design Systems Inc(CDNS)','Autozone Inc(AZO)','Valero Energy Corp(VLO)','Roper Technologies Inc(ROP)','Paychex Inc(PAYX)','Capital One Financial Corp(COF)','Republic Services Inc(RSG)','Archer-Daniels-Midland Company(ADM)','IQVIA Holdings Inc.(IQV)','Cintas Corp(CTAS)','Amphenol Corp(APH)','American International Group Inc(AIG)','Williams Inc(WMB)','Dollar Tree Inc(DLTR)','Phillips 66(PSX)','Xcel Energy Inc(XEL)','Kinder Morgan Inc(KMI)','Travelers Companies Inc(TRV)','Autodesk Inc(ADSK)','Freeport-McMoRan Inc.(FCX)','TE Connectivity Ltd(TEL)','Chipotle Mexican Grill Inc(CMG)','Motorola Solutions Inc(MSI)','Agilent Technologies Inc(A)','Digital Realty Trust Inc(DLR)','Aflac Inc(AFL)','SBA Communications Corporation(SBAC)','Electronic Arts Inc(EA)','Arthur J Gallagher(AJG)','Brown Forman Inc Class B(BF.B)','Kroger(KR)','Devon Energy Corp(DVN)','Prudential Financial Inc(PRU)','Microchip Technology Inc(MCHP)','Cognizant Technology Solutions(CTSH)','Yum Brands Inc(YUM)','Allstate Corp(ALL)','Consolidated Edison Inc(ED)','Bank Of New York Mellon Corp(BK)','Resmed Inc(RMD)','HP Inc(HPQ)','Johnson Controls International Plc(JCI)','Baxter International Inc(BAX)','Walgreen Boots Alliance Inc(WBA)','WEC Energy Group Inc(WEC)','Hilton Worldwide Holdings Inc(HLT)','Parker-Hannifin Corp(PH)','Biogen Inc(BIIB)','Global Payments Inc(GPN)','Simon Property Group Inc(SPG)','Idexx Laboratories Inc(IDXX)','Public Service Enterprise Group Inc(PEG)','Tyson Foods Inc Class A(TSN)','Hess Corp(HES)','AmerisourceBergen Corp(ABC)','Transdigm Group Inc(TDG)','International Flavors & Fragrances(IFF)','Nucor Corp(NUE)','Eversource Energy(ES)','Discover Financial Services(DFS)','Illumina Inc(ILMN)','Verisk Analytics Inc(VRSK)','Paccar Inc(PCAR)','Cummins Inc(CMI)','Lyondellbasell Industries NV(LYB)','Fastenal(FAST)','PPG Industries Inc(PPG)','Corning Inc(GLW)','M&T Bank Corp(MTB)','Ross Stores Inc(ROST)','American Water Works Inc(AWK)','AvalonBay Communities Inc(AVB)','Equity Residential(EQR)','Hormel Foods Corp(HRL)','Mettler Toledo Inc(MTD)','Weyerhaeuser Company(WY)','D R Horton Inc(DHI)','T Rowe Price Group Inc(TROW)','Ametek Inc(AME)','Halliburton(HAL)','Kellogg(K)','Ameriprise Finance Inc(AMP)','Oneok Inc(OKE)','DTE Energy(DTE)','Aptiv Plc(APTV)','Ebay Inc(EBAY)','ON Semiconductor Corp(ON)','Edison International(EIX)','W.W. Grainger Inc(GWW)','Church And Dwight Inc(CHD)','Rockwell Automation Inc(ROK)','Equifax Inc(EFX)','Albemarle Corp(ALB)','Southwest Airlines(LUV)','Tractor Supply(TSCO)','Extra Space Storage Inc(EXR)','Ameren Corp(AEE)','Alexandria Real Estate Equities Inc(ARE)','Entergy Corp(ETR)','Laboratory Corporation Of America(LH)','Lennar Corporation Class A(LEN)','Fifth Third Bancorp(FITB)','McCormick & Co  Non-voting(MKC)','State Street Corp(STT)','CDW Corp(CDW)','Duke Realty Corporation(DRE)','Firstenergy Corp(FE)','Zimmer Biomet Holdings Inc(ZBH)','Ansys Inc(ANSS)','Hartford Financial Services Group(HIG)','Ulta Beauty Inc(ULTA)','PPL Corporation(PPL)','Ventas Inc(VTR)','Waters Corp(WAT)','Northern Trust Corporation(NTRS)','Vulcan Materials(VMC)','Align Technology Inc(ALGN)','Martin Marietta Materials Inc(MLM)','Genuine Parts(GPC)','Fortive Corp(FTV)','Delta Air Lines Inc(DAL)','CMS Energy Corp(CMS)','Mid-America Apartment Communities Inc(MAA)','Verisign Inc(VRSN)','Gartner Inc(IT)','Garmin Ltd(GRMN)','Quanta Services Inc(PWR)','Centerpoint Energy Inc(CNP)','Raymond James Inc(RJF)','Clorox(CLX)','Cincinnati Financial Corp(CINF)','United Rentals Inc(URI)','Incyte Corp(INCY)','Fox Corporation Class B(FOX)','Fox Corporation Class A(FOXA)','VF Corp(VFC)','JB Hunt Transport Services Inc(JBHT)','Citizens Financial Group Inc(CFG)','Hologic Inc(HOLX)','Broadridge Financial Solutions Inc(BR)','Huntington Bancshares Inc(HBAN)','Essex Property Trust Inc(ESS)','Regions Financial Corp(RF)','Dover Corp(DOV)','Hewlett Packard Enterprise(HPE)','Perkinelmer Inc(PKI)','CF Industries Holdings Inc(CF)','Seagate Technology Plc(STX)','Ingersoll Rand Plc(IR)','Stanley Black & Decker Inc(SWK)','Best Buy Inc(BBY)','Skyworks Solutions Inc(SWKS)','Expeditors International Of Washington(EXPD)','Conagra Brands Inc(CAG)','Mosaic(MOS)','Quest Diagnostics Inc(DGX)','Keycorp(KEY)','International Paper(IP)','Principal Financial Group Inc(PFG)','Cooper Companies Inc.(COO)','Synchrony Financial(SYF)','Campbell Soup(CPB)','Cardinal Health Inc(CAH)','Nasdaq Inc(NDAQ)','Alliant Energy Corp(LNT)','Darden Restaurants Inc(DRI)','Marathon Oil Corp(MRO)','Western Digital Corp(WDC)','J.M. Smucker(SJM)','NetApp Inc(NTAP)','Carmax Inc(KMX)','LKQ Corp(LKQ)','UDR Inc(UDR)','Akamai Technologies Inc(AKAM)','AES Corp(AES)','Loews Corp(L)','Xylem Inc(XYL)','Expedia Inc(EXPE)','Boston Properties Inc(BXP)','Avery Dennison Corp(AVY)','Omnicom Group Inc(OMC)','Iron Mountain Incorporated(IRM)','Citrix Systems Inc(CTXS)','Textron Inc(TXT)','Packaging Corp Of America(PKG)','Molson Coors Brewing Class B(TAP)','FMC Corp(FMC)','Cboe Global Markets Inc.(CBOE)','Masco Corp(MAS)','United Continental Holdings Inc(UAL)','C.H. Robinson Worldwide Inc.(CHRW)','Franklin Resources Inc(BEN)','MGM Resorts International(MGM)','Kimco Realty Corporation(KIM)','Nisource Inc(NI)','Advance Auto Parts Inc(AAP)','Host Hotels & Resorts Inc(HST)','Eastman Chemical(EMN)','Hasbro Inc(HAS)','Interpublic Group Of Companies Inc(IPG)','Apache Corp(APA)','Snap On Inc(SNA)','Pultegroup Inc(PHM)','Henry Schein Inc(HSIC)','Everest Re Group Ltd(RE)','Qorvo Inc.(QRVO)','Regency Centers Corporation(REG)','Westrock(WRK)','Carnival Corp(CCL)','Comerica Inc(CMA)','American Airlines Group Inc(AAL)','Assurant Inc(AIZ)','Juniper Networks Inc(JNPR)','Dish Network Corp Class A(DISH)','News Corp Class B(NWS)','News Corp Class A(NWSA)','Whirlpool Corp(WHR)','F5 Networks Inc(FFIV)','A O Smith Corp(AOS)','NRG Energy Inc(NRG)','Universal Health Services Inc.(UHS)','Allegion PLC(ALLE)','Robert Half(RHI)','Nielsen Holdings Plc(NLSN)','Sealed Air Corp(SEE)','Huntington Ingalls Industries Inc(HII)','Fortune Brands Home And Security(FBHS)','Royal Caribbean Cruises Ltd(RCL)','Pinnacle West Corp(PNW)','Mohawk Industries Inc(MHK)','BorgWarner Inc(BWA)','Lincoln National Corp(LNC)','Tapestry(TPR)','Newell Brands Inc(NWL)','Davita Inc(DVA)','Federal Realty Investment Trust(FRT)','Pentair(PNR)','Dentsply Sirona Inc(XRAY)','Zions Bancorporation(ZION)','Invesco Ltd(IVZ)','Ralph Lauren Corp Class A(RL)','DXC Technology Company(DXC)','Wynn Resorts Ltd(WYNN)','Vornado Realty Trust(VNO)','Alaska Air Group Inc(ALK)','PVH Corp(PVH)','Norwegian Cruise Line Holdings Ltd(NCLH)'))
    
    # st.warning(option)

    str1 = "Dataset S&P/individual_stocks_5yr/individual_stocks_5yr/"

    str2 = option.split('(')[1].split(')')[0] + '_data.csv'

    # st.warning(str1+str2)

    df_stock = pd.read_csv(str1+str2)

    st.sidebar.text("Training Date Range :")

    start = st.sidebar.date_input("Training Date Start :",
                                    min_value= datetime.strptime(df_stock.date[0], '%Y-%m-%d'),
                                    max_value= datetime.strptime(df_stock.date[len(df_stock)-529], '%Y-%m-%d'),
                                    value = datetime.strptime(df_stock.date[0], '%Y-%m-%d'))
    end = st.sidebar.date_input("Training Date End :",
                                    min_value= start + timedelta(days=365),
                                    max_value= datetime.strptime(df_stock.date[len(df_stock)-164], '%Y-%m-%d'),
                                    value = datetime.strptime(df_stock.date[len(df_stock)-164], '%Y-%m-%d'))

    period = st.slider("Forecast Period",
                        min_value= 1,
                        value=164,
                        max_value= 365)

df_stock = df_stock.set_index('date', drop=True)
df_stock.index = pd.to_datetime(df_stock.index)
df_stock_train = df_stock[start:end]

if len(df_stock[end:]) >= period:

    df_stock_test = (df_stock[end:])[:period]

else :
    initial_df_stock_test = pd.DataFrame(df_stock[end+ timedelta(days=1):])

    extra_index = pd.date_range(start = initial_df_stock_test.index[len(initial_df_stock_test)-1] + timedelta(days=1), periods=(period-len(initial_df_stock_test)))
    extra_df_stock_test = pd.DataFrame(data=None, index = extra_index)

    df_stock_test = initial_df_stock_test.append(extra_df_stock_test)


# st.warning(period)

with st.sidebar:

    ARIMA, LSTM = st.columns(2)
   
    ARIMA_Model = ARIMA.checkbox("ARIMA")  

    LSTM_Model = LSTM.checkbox("LSTM")

    Prophet_M, VAR_M = st.columns(2)

    Prophet_Model = Prophet_M.checkbox("Facebook Prophet")

    VAR_Model = VAR_M.checkbox("Vector AutoRegressor (VAR)")

    Fifth_Model, GluonTs_Model, Sixth_Model = st.columns(3)

    GluonTs_Model.checkbox("Amazon GluonTs", disabled = True)


if VAR_Model:

    stock = df_stock.copy() 

    stock_train = df_stock_train.copy()
    stock_test = df_stock_test.copy()

    data_is_non_stationary = True

    differenced_stock_open = stock_train.open
    differenced_stock_close = stock_train.close
    differenced_stock_low = stock_train.low
    differenced_stock_high = stock_train.high
    differenced_stock_vol = stock_train.volume

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

    var_model_instance = VAR(var_stock)

    order_for_var = var_model_instance.select_order(40).bic

    var_result = var_model_instance.fit(order_for_var)

    print(var_result.summary())

    lag = var_result.k_ar


    stock_train = stock_train.drop(columns=['Name', 'volume'], axis=1)
    stock_test = stock_test.drop(columns=['Name', 'volume'], axis=1)
    
    var_model_instance = VARMAX(stock_train, order=(lag,0),enforce_stationarity= True)

    fitted_model = var_model_instance.fit(disp=True)

    result_forecast2 = fitted_model.get_prediction(start=len(stock_train)+1 , end=len(stock_train) + len(stock_test))

    VAR_prediction = result_forecast2.predicted_mean

    dateIndex = pd.DatetimeIndex(stock_test.index[:len(VAR_prediction)])

    VAR_prediction_df = (pd.DataFrame(VAR_prediction)).set_index(dateIndex)


    if len(stock) > (len(stock_train) + len(stock_test)):
        var_df = pd.DataFrame(stock.close)
        var_df['yhat'] = VAR_prediction_df.close
        st.line_chart(var_df, use_container_width=True)
        st.dataframe(var_df)

    else:
        var_df = pd.DataFrame(stock_train.close).append(pd.DataFrame(stock_test.close))
        var_df['yhat'] = VAR_prediction_df.close
        st.line_chart(var_df, use_container_width=True)
        st.dataframe(var_df)

    #mae_test, mae = st.columns(2)
    #mae_test.text('Mean Absolute Error is:')
    #mae.write(calc_mae(stock.close[-len(var_df):].values, VAR_prediction_df.close.values))

    #mse_test, mse = st.columns(2)
    #mse_test.text('Mean Square Error is:')
    #mse.write(calc_mse(stock.close[-len(var_df):].values, VAR_prediction_df.close.values))

    #rmse_test, rmse = st.columns(2)
    #rmse_test.text('Root Mean Absolute Error is:')
    #rmse.write(calc_rmse(stock.close[-len(var_df):].values, VAR_prediction_df.close.values))

    st.write('\nModel Metrices are as follows:')

    test_data_last_index = stock_test.index[len(stock_test)-1]
    test_data_length_available = len(stock.close[stock_test.index[0]:test_data_last_index])

    mae = calc_mae(stock.close[stock_test.index[0]:test_data_last_index].values, VAR_prediction_df.close[:test_data_length_available].values)
    mse = calc_mse(stock.close[stock_test.index[0]:test_data_last_index].values, VAR_prediction_df.close[:test_data_length_available].values)
    rmse = calc_rmse(stock.close[stock_test.index[0]:test_data_last_index].values, VAR_prediction_df[:test_data_length_available].close.values)

    display_metrices(mae, mse, rmse)
    
if LSTM_Model :
        st.markdown(body="# Long Short Term Memory Model:")

        stock = df_stock.copy()

        stock_train = df_stock_train.copy()
        stock_test = df_stock_test.copy()

        stock_lstm = stock[['close']]

        features = stock_train[['close']] 
        target = stock_train['close'].tolist() 
  

        #x_train, x_test, y_train, y_test = train_test_split(features,target, test_size=len(stock_test))

        window_length = 3
        feature_count = 1

        train_generator = TimeseriesGenerator(features, target, length=window_length, sampling_rate=1, batch_size=1, stride = 1)
        #test_generator = TimeseriesGenerator(x_test, y_test, length=window_length, sampling_rate=1, batch_size=1, stride = 1)

        LSTM_Model_Instance = keras.Sequential([
            keras.Input(shape=(window_length, feature_count)),
            layers.LSTM(128, activation = 'relu', return_sequences = True),
            layers.LSTM(128, activation = 'relu', return_sequences = True),
            layers.LSTM(64, activation = 'relu', return_sequences = False),
            layers.Dense(1)])

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                         patience = 3,
                                                         mode=min)
        LSTM_Model_Instance.compile(loss=tf.losses.MeanSquaredError(),
             optimizer = tf.optimizers.Adam(),
             metrics = [tf.metrics.MeanAbsoluteError()])
       
        history = LSTM_Model_Instance.fit_generator(train_generator, epochs=50,
                             #validation_data=test_generator,
                             shuffle=False, callbacks=[early_stopping])
        
        #LSTM_Model_Instance.evaluate_generator(test_generator, verbose=0)

        #LSTM_prediction = LSTM_Model_Instance.predict_generator(test_generator)

        
        LSTM_prediction = []

        first_eval_batch = stock_train[-window_length:].close.to_numpy()
        current_batch = first_eval_batch.reshape((1, window_length, feature_count))

        for i in range(len(stock_test)):
    
            # get the prediction value for the first batch
            current_pred = LSTM_Model_Instance.predict(current_batch)[0]
    
            # append the prediction into the array
            LSTM_prediction.append(current_pred) 
    
            # use the prediction to update the batch and remove the first value
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)    



        #dateIndex = pd.DatetimeIndex(stock_test.index[:len(LSTM_prediction)])
        dateIndex = pd.DatetimeIndex(stock_test.index)

        LSTM_prediction_df = (pd.DataFrame(LSTM_prediction)).set_index(dateIndex)

        if len(stock) > (len(stock_train) + len(stock_test)):
            lstm_df = pd.DataFrame(stock_lstm)
            lstm_df['yhat'] = LSTM_prediction_df
            st.line_chart(lstm_df, use_container_width=True)
            st.dataframe(lstm_df)

        else:
            lstm_df = pd.DataFrame(stock_train.close).append(pd.DataFrame(stock_test.close))
            lstm_df['yhat'] = LSTM_prediction_df
            st.line_chart(lstm_df, use_container_width=True)
            st.dataframe(lstm_df)

        #mae_test, mae = st.columns(2)
        #mae_test.text('Mean Absolute Error is:')
        #mae.write(calc_mae(stock.close[-len(stock_train):].values, LSTM_prediction))

        #mse_test, mse = st.columns(2)
        #mse_test.text('Mean Square Error is:')
        #mse.write(calc_mse(stock.close[-len(stock_train):].values, LSTM_prediction))

        #rmse_test, rmse = st.columns(2)
        #rmse_test.text('Root Mean Absolute Error is:')
        #rmse.write(calc_rmse(stock.close[-len(stock_train):].values, LSTM_prediction))

        st.write('\nModel Metrices are as follows:')

        test_data_last_index = stock_test.index[len(stock_test)-1]
        test_data_length_available = len(stock.close[stock_test.index[0]:test_data_last_index])

        mae = calc_mae(stock.close[stock_test.index[0]:test_data_last_index].values, LSTM_prediction_df[:test_data_length_available].values)
        mse = calc_mse(stock.close[stock_test.index[0]:test_data_last_index].values, LSTM_prediction_df[:test_data_length_available].values)
        rmse = calc_rmse(stock.close[stock_test.index[0]:test_data_last_index].values, LSTM_prediction_df[:test_data_length_available].values)

        display_metrices(mae, mse, rmse)

if ARIMA_Model :
    st.markdown(body="## AutoRegression Integrated Moving Average Model :")

    stock = df_stock.copy()

    stock_train = df_stock_train.copy().close
    stock_test = df_stock_test.copy().close

    Arima_Model_Instance = pm.auto_arima(stock_train, start_p = 0,  start_q=0, max_order=4, test='adf',
                            error_action='ignore', suppress_warning = True, stepwise = True, trace = True)

    prediction_Arima = Arima_Model_Instance.predict(len(stock_test), return_conf_int=True)

    ARIMA_prediction_df = pd.DataFrame(prediction_Arima[0])
    
    dateIndex = pd.DatetimeIndex(stock_test.index)

    ARIMA_prediction_df = ARIMA_prediction_df.set_index(dateIndex)

    if len(stock) > (len(stock_train) + len(stock_test)):
        arima_df = pd.DataFrame(stock.close)
        arima_df['yhat'] = ARIMA_prediction_df
        st.line_chart(arima_df, use_container_width=True)
        st.dataframe(arima_df)

    else:
        arima_df = pd.DataFrame(stock_train).append(pd.DataFrame(stock_test))
        arima_df['yhat'] = ARIMA_prediction_df
        st.line_chart(arima_df, use_container_width=True)
        st.dataframe(arima_df)



    #mae_test, mae = st.columns(2)
    #mae_test.text('Mean Absolute Error is:')
    #mae.write(calc_mae(stock.close[-len(arima_df):].values, ARIMA_prediction_df.values))

    #mse_test, mse = st.columns(2)
    #mse_test.text('Mean Square Error is:')
    #mse.write(calc_mse(stock.close[-len(arima_df):].values, ARIMA_prediction_df.values))

    #rmse_test, rmse = st.columns(2)
    #rmse_test.text('Root Mean Absolute Error is:')
    #rmse.write(calc_rmse(stock.close[-len(arima_df):].values, ARIMA_prediction_df.values    ))

    st.write('\nModel Metrices are as follows:')
    test_data_length_available = len(stock.close[stock_test.index[0]:])
    test_data_last_index = stock_test.index[len(stock_test)-1]

    mae = calc_mae(stock.close[stock_test.index[0]:test_data_last_index].values, ARIMA_prediction_df[:test_data_length_available].values)
    mse = calc_mse(stock.close[stock_test.index[0]:test_data_last_index].values, ARIMA_prediction_df[:test_data_length_available].values)
    rmse = calc_rmse(stock.close[stock_test.index[0]:test_data_last_index].values, ARIMA_prediction_df[:test_data_length_available].values)

    display_metrices(mae, mse, rmse)


if Prophet_Model:

    st.markdown(body="## Facebook Prophet :")

    stock = df_stock.copy()

    stock_train = df_stock_train.copy().close
    stock_test = df_stock_test.copy().close

    stock_prophet = stock_train.reset_index()[['date','close']].rename({'date':'ds','close':'y'},axis='columns')

    prophet_model_instance = Prophet() #By Default confidence interval_width is 80, interval_width=.95

    prophet_model_instance.fit(stock_prophet)

    future_dataframe = prophet_model_instance.make_future_dataframe(periods=len(stock_test))

    forecasted_data = prophet_model_instance.predict(future_dataframe)[len(stock_train):]

    forecasted_data_df = pd.DataFrame(forecasted_data.yhat)
    dateIndex = pd.DatetimeIndex(stock_test.index)
    Prophet_prediction_df = forecasted_data_df.set_index(dateIndex)

    if len(stock) > (len(stock_train) + len(stock_test)):
        prophet_df = pd.DataFrame(stock.close)
        prophet_df['yhat'] = Prophet_prediction_df.yhat
        st.line_chart(prophet_df, use_container_width=True)
        st.dataframe(prophet_df)

    else:
        prophet_df = pd.DataFrame(stock_train).append(pd.DataFrame(stock_test))
        prophet_df['yhat'] = Prophet_prediction_df
        st.line_chart(prophet_df, use_container_width=True)
        st.dataframe(prophet_df)

    

    #mae_test, mae = st.columns(2)
    #mae_test.text('Mean Absolute Error is:')
    #mae.write(calc_mae(stock.close[-len(prophet_df):].values, Prophet_prediction_df.values))

    #mse_test, mse = st.columns(2)
    #mse_test.text('Mean Square Error is:')
    #mse.write(calc_mse(stock.close[-len(prophet_df):].values, Prophet_prediction_df.values))

    #rmse_test, rmse = st.columns(2)
    #rmse_test.text('Root Mean Absolute Error is:')
    #rmse.write(calc_rmse(stock.close[-len(prophet_df):].values, Prophet_prediction_df.values))
    st.write('\nModel Metrices are as follows:')
    test_data_length_available = len(stock.close[stock_test.index[0]:])
    test_data_last_index = stock_test.index[len(stock_test)-1]

    mae = calc_mae(stock.close[stock_test.index[0]:test_data_last_index].values, Prophet_prediction_df[:test_data_length_available].values)
    mse = calc_mse(stock.close[stock_test.index[0]:test_data_last_index].values, Prophet_prediction_df[:test_data_length_available].values)
    rmse = calc_rmse(stock.close[stock_test.index[0]:test_data_last_index].values, Prophet_prediction_df[:test_data_length_available].values)

    display_metrices(mae, mse, rmse)



