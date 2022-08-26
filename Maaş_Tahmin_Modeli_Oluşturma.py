# Makine Öğrenmesi ile Maaş Tahmini

# İş Problemi

"""
Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
oyuncularının maaş tahminleri için bir makine öğrenmesi modeli geliştiriniz.

"""

# Veri Seti Hikayesi

"""
Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır. 
Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve kariyer istatistikleri, 
Collier Books, Macmillan Publishing Company,
New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.
 
"""

# Değişkenler

""" 
Değişkenler

20 Değişken 322 Gözlem 21 KB

AtBat     1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
Hits      1986-1987 sezonundaki isabet sayısı
HmRun     1986-1987 sezonundaki en değerli vuruş sayısı
Runs      1986-1987 sezonunda takımına kazandırdığı sayı
RBI       Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
Walks     Karşı oyuncuya yaptırılan hata sayısı
Years     Oyuncunun major liginde oynama süresi (sene)
CAtBat    Oyuncunun kariyeri boyunca topa vurma sayısı
CHits     Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
CHmRun    Oyucunun kariyeri boyunca yaptığı en değerli sayısı
CRuns     Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
CRBI      Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
CWalks    Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
League    Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
Division  1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
PutOuts   Oyun icinde takım arkadaşınla yardımlaşma
Assits    1986-1987 sezonunda oyuncunun yaptığı asist sayısı
Errors    1986-1987 sezonundaki oyuncunun hata sayısı
Salary    Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
NewLeague 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

"""
# Görev

# Veriönişleme,
# Özellikmühendisliği
# işlemleri gerçekleştirerek maaş tahmin modeli geliştiriniz.

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

data=pd.read_csv("dsmlbc_9_abdulkadir/Homeworks/hakan_erdem/6_Makine_Ogrenmesi_YapayOgrenme/Maaş Tahmin Modeli Oluşturma/hitters.csv")
df=data.copy()

# Keşifçi Veri Analizi
# Genel Resim

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

""""
Variables: 20
cat_cols: 3
num_cols: 17
cat_but_car: 0
num_but_cat: 0

"""

# Kategorik Değişken Analizi (Analysis of Categorical Variables)

def cat_summary(dataframe, col_name, plot=False):

    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

"""
   League  Ratio
A     175 54.348
N     147 45.652
##########################################
   Division  Ratio
W       165 51.242
E       157 48.758
##########################################
   NewLeague  Ratio
A        176 54.658
N        146 45.342
##########################################

"""

# Sayısal Değişken Analizi (Analysis of Numerical Variables)

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# Hedef Değişken Analizi (Analysis of Target Variable)

def target_summary_with_cat(dataframe, target, categorical_col):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

"""
        TARGET_MEAN
League             
A           542.000
N           529.118
          TARGET_MEAN
Division             
E             624.271
W             450.877
           TARGET_MEAN
NewLeague             
A              537.113
N              534.554

"""

# Korelasyon Analizi (Analysis of Correlation)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True)

"""
 ['Hits', 'Runs', 'CAtBat', 'CHits', 'CRuns', 'CRBI', 'CWalks']
 
"""

# Veri Ön İşleme
# Eksik Değer Analizi

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df, na_name=True)

"""
        n_miss  ratio
Salary      59 18.320
Out[5]: ['Salary']

"""

df.dropna(inplace=True)

"""
Empty DataFrame
Columns: [n_miss, ratio]
Index: []
Out[7]: []

"""
# Aykırı Değer Analizi

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

"""
AtBat False
Hits False
HmRun True ***
Runs False
RBI False
Walks False
Years True ***
CAtBat True ***
CHits True ***
CHmRun True ***
CRuns True ***
CRBI True *** 
CWalks True ***
PutOuts True ***
Assists True *** 
Errors True *** 
Salary True ***

"""

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

"""
AtBat False
Hits False
HmRun False
Runs False
RBI False
Walks False
Years False
CAtBat False
CHits False
CHmRun False
CRuns False
CRBI False
CWalks False
PutOuts False
Assists False
Errors False
Salary False

"""

# Özellik Çıkarımı

new_num_cols=[col for col in num_cols if col!="Salary"]
df.head()
df[new_num_cols]=df[new_num_cols]+0.0000000001

df['NEW_Hits'] = df['Hits'] / df['CHits'] + df['Hits']
df['NEW_RBI'] = df['RBI'] / df['CRBI']
df['NEW_Walks'] = df['Walks'] / df['CWalks']
df['NEW_PutOuts'] = df['PutOuts'] * df['Years']
df["Hits_Success"] = (df["Hits"] / df["AtBat"]) * 100
df["NEW_CRBI*CATBAT"] = df['CRBI'] * df['CAtBat']
df["NEW_RBI"] = df["RBI"] / df["CRBI"]
df["NEW_Chits"] = df["CHits"] / df["Years"]
df["NEW_CHmRun"] = df["CHmRun"] * df["Years"]
df["NEW_CRuns"] = df["CRuns"] / df["Years"]
df["NEW_Chits"] = df["CHits"] * df["Years"]
df["NEW_RW"] = df["RBI"] * df["Walks"]
df["NEW_RBWALK"] = df["RBI"] / df["Walks"]
df["NEW_CH_CB"] = df["CHits"] / df["CAtBat"]
df["NEW_CHm_CAT"] = df["CHmRun"] / df["CAtBat"]
df['NEW_Diff_Atbat'] = df['AtBat'] - (df['CAtBat'] / df['Years'])
df['NEW_Diff_Hits'] = df['Hits'] - (df['CHits'] / df['Years'])
df['NEW_Diff_HmRun'] = df['HmRun'] - (df['CHmRun'] / df['Years'])
df['NEW_Diff_Runs'] = df['Runs'] - (df['CRuns'] / df['Years'])
df['NEW_Diff_RBI'] = df['RBI'] - (df['CRBI'] / df['Years'])
df['NEW_Diff_Walks'] = df['Walks'] - (df['CWalks'] / df['Years'])
df.head()

# One-Hot Encoding

# cat_cols  = 'League', 'Division', 'NewLeague'

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

"""
Out[8]: 
    AtBat    Hits  HmRun   Runs    RBI  Walks  Years   CAtBat    CHits  CHmRun   CRuns    CRBI  CWalks  PutOuts  Assists  Errors  Salary  NEW_Hits  NEW_RBI  NEW_Walks  \
1 315.000  81.000  7.000 24.000 38.000 39.000 14.000 3449.000  835.000  69.000 321.000 414.000 375.000  632.000   43.000  10.000 475.000    81.097    0.092      0.104   
2 479.000 130.000 18.000 66.000 72.000 76.000  3.000 1624.000  457.000  63.000 224.000 266.000 263.000  636.000   82.000  14.000 480.000   130.284    0.271      0.289   
3 496.000 141.000 20.000 65.000 78.000 37.000 11.000 5628.000 1575.000 208.750 828.000 838.000 354.000  200.000   11.000   3.000 500.000   141.090    0.093      0.105   
4 321.000  87.000 10.000 39.000 42.000 30.000  2.000  396.000  101.000  12.000  48.000  46.000  33.000  636.000   40.000   4.000  91.500    87.861    0.913      0.909   
5 594.000 169.000  4.000 74.000 51.000 35.000 11.000 4408.000 1133.000  19.000 501.000 336.000 194.000  282.000  421.000  25.000 750.000   169.149    0.152      0.180   

   NEW_PutOuts  Hits_Success  NEW_CRBI*CATBAT  NEW_Chits  NEW_CHmRun  NEW_CRuns   NEW_RW  NEW_RBWALK  NEW_CH_CB  NEW_CHm_CAT  NEW_Diff_Atbat  NEW_Diff_Hits  \
1     8848.000        25.714      1427886.000  11690.000     966.000     22.929 1482.000       0.974      0.242        0.020          68.643         21.357   
2     1908.000        27.140       431984.000   1371.000     189.000     74.667 5472.000       0.947      0.281        0.039         -62.333        -22.333   
3     2200.000        28.427      4716264.000  17325.000    2296.250     75.273 2886.000       2.108      0.280        0.037         -15.636         -2.182   
4     1272.000        27.103        18216.000    202.000      24.000     24.000 1260.000       1.400      0.255        0.030         123.000         36.500   
5     3102.000        28.451      1481088.000  12463.000     209.000     45.545 1785.000       1.457      0.257        0.004         193.273         66.000   
 
   NEW_Diff_HmRun  NEW_Diff_Runs  NEW_Diff_RBI  NEW_Diff_Walks  League_N  Division_W  NewLeague_N  
1           2.071          1.071         8.429          12.214         1           1            1  
2          -3.000         -8.667       -16.667         -11.667         0           1            0  
3           1.023        -10.273         1.818           4.818         1           0            1  
4           4.000         15.000        19.000          13.500         1           0            1  
5           2.273         28.455        20.455          17.364         0           1            0  

"""

# Özellik Ölçeklendirme

cat_cols, num_cols, cat_but_car = grab_col_names(df)
"""
Observations: 263
Variables: 39
cat_cols: 3
num_cols: 36
cat_but_car: 0
num_but_cat: 3

"""

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

"""
   AtBat   Hits  HmRun   Runs    RBI  Walks  Years  CAtBat  CHits  CHmRun  CRuns   CRBI  CWalks  PutOuts  Assists  Errors  Salary  NEW_Hits  NEW_RBI  NEW_Walks  \
1 -0.603 -0.596 -0.529 -1.206 -0.522 -0.098  1.426   0.375  0.201   0.101 -0.108  0.373   0.617    2.114   -0.524   0.218 475.000    -0.601   -0.836     -0.795   
2  0.513  0.492  0.734  0.442  0.794  1.609 -0.910  -0.461 -0.419   0.005 -0.425 -0.166   0.093    2.136   -0.253   0.831 480.000     0.492   -0.231     -0.151   
3  0.628  0.736  0.963  0.402  1.026 -0.190  0.789   1.373  1.415   2.339  1.547  1.920   0.519   -0.298   -0.746  -0.853 500.000     0.732   -0.832     -0.793   
4 -0.562 -0.462 -0.185 -0.618 -0.367 -0.513 -1.123  -1.023 -1.002  -0.811 -1.000 -0.969  -0.982    2.136   -0.545  -0.700  91.500    -0.451    1.943      2.009   
5  1.295  1.358 -0.874  0.755 -0.019 -0.282  0.789   0.814  0.690  -0.699  0.479  0.089  -0.229    0.160    2.099   2.514 750.000     1.355   -0.633     -0.529   
 
   NEW_PutOuts  Hits_Success  NEW_CRBI*CATBAT  NEW_Chits  NEW_CHmRun  NEW_CRuns  NEW_RW  NEW_RBWALK  NEW_CH_CB  NEW_CHm_CAT  NEW_Diff_Atbat  NEW_Diff_Hits  \
1        3.414        -0.168            0.016      0.381       0.297     -1.063  -0.466      -0.580     -0.874       -0.277          -0.001          0.095   
2        0.047         0.277           -0.470     -0.623      -0.502      1.427   1.485      -0.612      0.805        1.005          -1.142         -1.220   
3        0.188         0.679            1.621      0.929       1.664      1.456   0.220       0.781      0.738        0.889          -0.735         -0.614   
4       -0.262         0.265           -0.672     -0.736      -0.671     -1.012  -0.575      -0.069     -0.321        0.426           0.472          0.550   
5        0.626         0.686            0.042      0.456      -0.481      0.025  -0.318      -0.000     -0.236       -1.347           1.083          1.438   
  
   NEW_Diff_HmRun  NEW_Diff_Runs  NEW_Diff_RBI  NEW_Diff_Walks  League_N  Division_W  NewLeague_N  
1          -0.274         -0.476        -0.215           0.145         1           1            1  
2          -1.273         -1.011        -1.672          -1.504         0           1            0  
3          -0.481         -1.099        -0.599          -0.366         1           0            1  
4           0.105          0.290         0.399           0.233         1           0            1  
5          -0.235          1.030         0.484           0.500         0           1            0  

"""

# Modelleme

# Base Modeller

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

"""
RMSE: 234.3408 (LR) 
RMSE: 231.6527 (Ridge) 
RMSE: 228.3764 (Lasso) 
RMSE: 256.4444 (ElasticNet) 
RMSE: 255.7111 (KNN) 
RMSE: 254.2879 (CART) 
RMSE: 204.7818 (RF) 
RMSE: 398.358 (SVR) 
RMSE: 206.2825 (GBM) 
RMSE: 224.8234 (XGBoost) 
RMSE: 225.011 (LightGBM) 
RMSE: 218.8434 (CatBoost) 

"""

# Model Optimizasyonu ve Başarı Değerlendirme

# Random Forest

rf_model = RandomForestRegressor(random_state=17)
rf_params = {"max_depth": [5, 8, 15, None],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

"""
203.25571066107042

"""

# GBM Model

gbm_model = GradientBoostingRegressor(random_state=17)
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

"""
199.27635401704464
"""

# LightGBM

lgbm_model = LGBMRegressor(random_state=17)
lgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [300, 500],
                "colsample_bytree": [0.7, 1]}
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

"""
220.9842866192677
"""

# CatBoost

catboost_model = CatBoostRegressor(random_state=17, verbose=False)
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(catboost_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

"""
211.38250877309156
"""

SONUÇ: En iyi sonucu rmse = 199.27635401704464 değeri ile GBM modelinden elde ettik.





