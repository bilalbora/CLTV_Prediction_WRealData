##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması


##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Değişkenler

# master_id: Müşterinin eşsiz id numarası
# order_channel: Siparişin verildiği kanal
# last_order_channel: Son siparişin verildiği kanal
# first_order_date: İlk sipariş tarihi
# last_order_date: Son sipariş tarihi
# last_order_date_online: Online kanaldan verilen son sipariş
# last_order_date_offline: Offline kanaldan verilen son sipariş
# order_num_total_ever_online: Online'dan verilen sipariş toplamı
# order_num_total_ever_offline: Offline'dan verilen sipariş toplamı
# customer_value_total_ever_offline: Müşterinin offlinedan verdiği siparişin değeri
# customer_value_total_ever_online: Müşterinin onlinedan verdiği siparişin değeri
# interested_in_categories_12: Müşterinin ilgilendiği kategoriler

###################################
# Gerekli Kütüphane ve Fonksiyonlar
###################################

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#################################
# Verinin Okunması ve İncelenmesi
#################################

df_ = pd.read_csv('flo_data_20k.csv')
df = df_.copy()
df.head()

df.isnull().sum()

df.dtypes
# Çıktıya baktığımızda tarih ile ilgili olan değişkenler object tipinde. Bunları datetime'a dönüştürmemiz gerekir.

date_variables = ['first_order_date', 'last_order_date', 'last_order_date_online', 'last_order_date_offline']

def to_datetime(dataframe, columns):
    for i in columns:
        dataframe[i] = pd.to_datetime(dataframe[i])
    return dataframe.dtypes

to_datetime(df,date_variables)

# Yeni değişkenler oluşturmadan önce aykırı değerlerin kontrolü

df.describe().T

out_variables = ['order_num_total_ever_online', 'order_num_total_ever_offline',
                 'customer_value_total_ever_offline', 'customer_value_total_ever_online']

for i in out_variables:
    replace_with_thresholds(df, i)

df.describe().T

# Siparişler üzerinden müşterilere ait toplam harcalamarı ve sipariş başına miktarları bulmak

df['spend_per_order_off'] = df['customer_value_total_ever_offline'] / df['order_num_total_ever_offline']
df['spend_per_order_on'] = df['customer_value_total_ever_online'] / df['order_num_total_ever_online']
df['spend_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df['order_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']

# Verilen siparişlerde ondalıklı olamayacağı için integer tipine dönüştürülür
df['order_num_total_ever_online'] = df['order_num_total_ever_online'].astype('int')
df['order_num_total_ever_offline'] = df['order_num_total_ever_offline'].astype('int')
df['order_total'] = df['order_total'].astype('int')

######################################
# Lifetime Veri Yapısının Hazırlanması
######################################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

last_date = df['last_order_date_online'].max()
today_date = dt.datetime(2021, 6, 2)

cltv_df = pd.DataFrame()
cltv_df['customer_id'] = df['master_id']
cltv_df['recency_week'] = ((df['last_order_date'] - df['first_order_date']).astype('timedelta64[D]')) / 7
cltv_df['T_week'] = ((today_date - df['first_order_date']).astype('timedelta64[D]')) / 7
cltv_df['frequency'] = df['order_total']
cltv_df['monetary'] = df['spend_total'] / df['order_total']

###############################
# 2. BG-NBD Modelinin Kurulması
###############################
bgf = BetaGeoFitter(penalizer_coef=0.001)

def find_weekly_expect_purc(dataframe, week, plot=False):

    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_week'],
            cltv_df['T_week'])

    dataframe['expected_purc_' + str(week) + '_week'] = bgf.predict(week,
                                                                    dataframe['frequency'],
                                                                    dataframe['recency_week'],
                                                                    dataframe['T_week'])
    if plot:
        plot_period_transactions(bgf)
        plt.show()

    return dataframe

# 1, 6 ve 12 haftalık periyotlarda en çok satın alma beklenilen müşterileri bulalım


find_weekly_expect_purc(cltv_df, 1)
find_weekly_expect_purc(cltv_df, 6)
find_weekly_expect_purc(cltv_df, 12,plot=True)

cltv_df["expected_purc_1_week"].sort_values(ascending=False).head(10)

cltv_df["expected_purc_6_week"].sort_values(ascending=False).head(10)

cltv_df["expected_purc_12_week"].sort_values(ascending=False).head(10)


####################################
# 3. GAMMA-GAMMA Modelinin Kurulması
####################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

def set_gamma_gammamodel(dataframe):

    dataframe["expected_average_profit"] = ggf.conditional_expected_average_profit(dataframe['frequency'],
                                                                                 dataframe['monetary'])

set_gamma_gammamodel(cltv_df)


###################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
###################################################

def calc_cltv(dataframe, montly_time):

    cltv = ggf.customer_lifetime_value(bgf,
                                       dataframe['frequency'],
                                       dataframe['recency_week'],
                                       dataframe['T_week'],
                                       dataframe['monetary'],
                                       time=montly_time,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    dataframe['cltv'] = cltv
    return dataframe

cltv_final = calc_cltv(cltv_df, 6)
cltv_final.head()

# En değerli 20 kişi
cltv_final.sort_values(by='cltv', ascending=False).head(20)

############################################
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
############################################

def cltv_final_segment(dataframe, segmentsayisi):
    dataframe["segment"] = pd.qcut(dataframe["cltv"], segmentsayisi, labels=["D", "C", "B", "A"])
    return dataframe

cltv_final = cltv_final_segment(cltv_final, 4)

cltv_final.groupby('segment').agg({'cltv': ['mean', 'sum', 'count']})


# SONUÇ: Her bir müşteri için 6 aylık değer tahmini yapıldı ve en çok gelir getirmesi beklenen
# A segmenti müşterileri ve en az gelir getirmesi beklenen D segmenti müşterileri ayrıştırıldı.
# Bu sayede firma beklentilerine göre istediği gruba yönelip kampanyalar ortaya çıkarabilir.





















