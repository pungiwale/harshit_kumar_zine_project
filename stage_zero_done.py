"""
stage_zero.py

this program is to clean  a datasheet for model
development
"""
from sklearn.preprocessing import LabelEncoder
from statistics import mode
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
dataframes = []
print("tell me if you want to clean training data or testing data")
n=int(input("for training enter 4 or for testing enter 0"))
f=input("Enter the training file")
file=pd.read_csv(f,on_bad_lines='skip')
original_columns=file.columns
dataframes.append(file)
for i in range(n):
    f=input("enter next file name: ")
    df = pd.read_csv(f, on_bad_lines='skip')

    dataframes.append(df)
df = pd.concat(dataframes, axis=0)
df=df[original_columns]


if "attack_cat" in df.columns:
    df["attack_cat"] = df["attack_cat"].fillna("-")


df = df[df.isnull().sum(axis=1) < 20]
if "dur" in df.columns:
    df["dur"] = df["dur"].fillna(df["dur"].mean())

if "proto" in df.columns:
    df["proto"] = df["proto"].fillna(df["proto"].mode()[0])

if "service" in df.columns:
    df["service"] = df["service"].fillna("-")

if "state" in df.columns:
    df["state"] = df["state"].fillna("-")

if "sbytes" in df.columns:
    df["sbytes"] = df["sbytes"].fillna(df["sbytes"].mean())
if "dbytes" in df.columns:
    df["dbytes"] = df["dbytes"].fillna(df["dbytes"].mean())
if "rate" in df.columns:
    df["rate"] = df["rate"].fillna(df["rate"].mean())
if "sttl" in df.columns:
    df["sttl"] = df["sttl"].fillna(df["sttl"].mean())
if "dttl" in df.columns:
    df["dttl"] = df["dttl"].fillna(df["dttl"].mean())
if "spkts" in df.columns:
    df["spkts"].fillna(mode(df["spkts"]),inplace=True)
if "dpkts" in df.columns:
    df["dpkts"].fillna(mode(df["dpkts"]),inplace=True)

if "sload" in df.columns:
    df["sload"] = df["sload"].fillna(df["sload"].mean())

if "dload" in df.columns:
    df["dload"] = df["dload"].fillna(df["dload"].mean())

if "sloss" in df.columns:
    df["sloss"] = df["sloss"].fillna(df["sloss"].mean())

if "dloss" in df.columns:
    df["dloss"] = df["dloss"].fillna(df["dloss"].mean())

if "sinpkt" in df.columns:
    df["sinpkt"] = df["sinpkt"].fillna(df["sinpkt"].mean())

if "dinpkt" in df.columns:
    df["dinpkt"] = df["dinpkt"].fillna(df["dinpkt"].mean())

if "smeansz" in df.columns:
    df["smeansz"] = df["smeansz"].fillna(df["smeansz"].mean())
#df["smeansz"].fillna(mode(df["smeansz"]),inplace=True)
if "dmeansz" in df.columns:
    df["dmeansz"] = df["dmeansz"].fillna(df["dmeansz"].mean())
if "smean" in df.columns:
    df["smean"] = df["smean"].fillna(df["smean"].mean())
if "dmean" in df.columns:
    df["dmean"] = df["dmean"].fillna(df["dmean"].mean())
if "sjit" in df.columns:
    df["sjit"].fillna(mode(df["sjit"]),inplace=True)
if "swin" in df.columns:
    df["swin"].fillna(df["swin"].mean(),inplace=True)
if "stcpb" in df.columns:
    df["stcpb"].fillna(df["stcpb"].mean(),inplace=True)
if "dtcpb" in df.columns:
    df["dtcpb"].fillna(df["dtcpb"].mean(),inplace=True)
if "dwin" in df.columns:
    df["dwin"].fillna(df["dwin"].mean(),inplace=True)
if "tcprtt" in df.columns:
    df["tcprtt"].fillna(df["tcprtt"].mean(),inplace=True)
if "synack" in df.columns:
    df["synack"].fillna(df["synack"].mean(), inplace=True)
if "ackdat" in df.columns:
    df["ackdat"].fillna(df["ackdat"].mean(),inplace=True)
if "smean" in df.columns:
    df["smean"].fillna(df["smean"].mean(),inplace=True)
if "dmean" in df.columns:
    df["dmean"].fillna(df["dmean"].mean(),inplace=True)
if "trans_depth" in df.columns:
    df["trans_depth"].fillna(df["trans_depth"].mean(),inplace=True)
if "response_body_len" in df.columns:
    df["response_body_len"].fillna(df["response_body_len"].mode()[0],inplace=True)
if "ct_srv_src" in df.columns:
    df["ct_srv_src"].fillna(df["ct_srv_src"].mode()[0],inplace=True)
if "ct_state_ttl" in df.columns:
    df["ct_state_ttl"].fillna(df["ct_state_ttl"].mode()[0],inplace=True)
if "ct_dst_Itm" in df.columns:
    df["ct_dst_Itm"] = df["ct_dst_Itm"].fillna(df["ct_dst_Itm"].mean())
if "ct_src_dport_Itm" in df.columns:
    df["ct_src_dport_Itm"].fillna(df["ct_src_dport_Itm"].mode()[0],inplace=True)
if "ct_dst_sport_Itm" in df.columns:
    df["ct_dst_sport_Itm"].fillna(df["ct_dst_sport_Itm"].mode()[0],inplace=True)
if "ct_dst_src_Itm" in df.columns:
    df["ct_dst_src_Itm"].fillna(mode(df["ct_dst_src_Itm"]),inplace=True)
if "ct_flp_cmd" in df.columns:
    df["ct_flp_cmd"] = df["ct_flp_cmd"].fillna(df["ct_flp_cmd"].mode()[0])

if "ct_flw_http_mthd" in df.columns:
    df["ct_flw_http_mthd"].fillna(mode(df["ct_flw_http_mthd"]),inplace=True)
if "ct_src_Itm" in df.columns:
    df["ct_src_Itm"] = df["ct_src_Itm"].fillna(df["ct_src_Itm"].mode()[0])

if "ct_srv_dst" in df.columns:
    df["ct_srv_dst"] = df["ct_srv_dst"].fillna(df["ct_srv_dst"].mean())

if "is_sm_ips_ports" in df.columns:
    df["is_sm_ips_ports"] = df["is_sm_ips_ports"].fillna(df["is_sm_ips_ports"].mean())

if "label" in df.columns:
    df["label"] = df["label"].fillna(df["label"].mean())

if "is_ftp_login" in df.columns:
    df["is_ftp_login"] = df["is_ftp_login"].fillna(df["is_ftp_login"].mean())

if "sport" in df.columns:
    df["sport"] = df["sport"].fillna(df["sport"].mean())

if "dsport" in df.columns:
    df["dsport"] = df["dsport"].fillna(df["dsport"].mean())

if "dstip" in df.columns:
    df["dstip"] = df["dstip"].fillna(df["dstip"].mean())

if "srcip" in df.columns:
    df["srcip"] = df["srcip"].fillna(df["srcip"].mean())

if "djit" in df.columns:
    df["djit"] = df["djit"].fillna(df["djit"].mean())



nominal_columns = ['proto', 'service', 'state', 'attack_cat']
# Ensure the columns exist in the DataFrame
#nominal_columns = [col for col in nominal_columns if col in df.columns]
# Initialize OneHotEncoder
#ohe = OneHotEncoder(sparse_output=False, drop=None)
# Fit and transform the data

# Ensure the columns exist in the DataFrame
nominal_columns = [col for col in nominal_columns if col in df.columns]

# Initialize LabelEncoder
label_encoders = {}
for col in nominal_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
#encoded_ohe = ohe.fit_transform(df[nominal_columns])

# Create a DataFrame for the one-hot encoded data
#ohe_columns = ohe.get_feature_names_out(nominal_columns)  # Get new column names
#df_ohe = pd.DataFrame(encoded_ohe, columns=ohe_columns)


# Drop original nominal columns and concatenate the encoded data
#df = df.drop(columns=nominal_columns)
#df = pd.concat([df, df_ohe], axis=1)

column_name=["dur","sbytes","dbytes","sttl",
"dttl","sloss","dloss","sload","dload","Spkts",
"Dpkts","swin","dwin","smeansz","dmeansz",
"trans_depth","res_bdy_len","Sjit","Djit",
"Sintpkt","Dintpkt","tcprtt","synack","ackdat","dintpkt","sintpkt",
"ct_state_ttl","ct_flw_http_mthd","ct_ftp_cmd",
"ct_srv_src","ct_srv_dst","ct_dst_Itm","sjit","djit",
"ct_src_Itm","ct_src_dport_Itm","response_body_len",
"ct_dst_sport_Itm","ct_dst_src_Itm","sload","smean","dmean"]
for cname in column_name:
  # Example: Removing outliers using the interquartile range (IQR)
    if cname in df.columns:
       Q1 = df[cname].quantile(0.25)
       Q3 = df[cname].quantile(0.75)
       IQR = Q3 - Q1
       df = df[~((df[cname] < (Q1 - 1.5 * IQR)) | (df[cname] > (Q3 + 1.5 * IQR)))]

df = df.drop_duplicates()
ff=input("Enter new file name")
df.to_csv(ff, index=False)
