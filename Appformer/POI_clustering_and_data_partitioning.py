import pandas as pd
import tqdm
import os
import pickle
from kmodes.kmodes import KModes
from sklearn.preprocessing import StandardScaler

def dict2file(dic, filename):
    with open(filename, 'w') as f:
        for k, v in dic.items():
            f.write("{}\t{}\n".format(k, v))

def generate_pri_app2id(df_pri, idx):
    app_counts = df_pri['app'].value_counts()
    length = len(app_counts) - 1
    top_apps = app_counts.iloc[:length]
    top_apps = top_apps.axes[0].tolist()
    pri_app2id = {top_apps[i]: i+1 for i in range(len(top_apps))}
    dict2file(pri_app2id, "data/Tsinghua_new/time_division/" + idx + "/primary/pri_app2id.txt")
    return pri_app2id

def generate_primary_dataset(data, path, length, name):
    path = os.path.join(path, str(length))
    if not os.path.exists(path):
        os.makedirs(path)
    prev_user = -1

    app_seq = []
    time_seq = []
    location_seq = []

    all_app_seq = []
    all_time_seq = []
    all_user_seq = []
    all_location_seq = []

    seq_length = length
    for i in tqdm.tqdm(range(len(data))):
        user = data.iloc[i]['user']
        location = data.iloc[i]['location_vectors']
        app = data.iloc[i]['app']
        time = data.iloc[i]['date']

        # 如果下一条数据为新的user，直接重新开始累计数据, 并丢弃已经加载的seq
        if prev_user != user:
            app_seq = [app]
            time_seq = [time]
            location_seq = [location]

        else:
            # 如果序列长度累计到所需长度，则添加到数据集中
            if len(app_seq) == seq_length:
                all_user_seq.append(user)
                all_app_seq.append(app_seq)
                all_time_seq.append(time_seq)
                all_location_seq.append(location_seq)

                # 构建好一个样本时在下一次迭代前需要把时间最早的数据剔除
                app_seq = app_seq[1:] + [app]
                time_seq = time_seq[1:] + [time]
                location_seq = location_seq[1:] + [location]
            else:
                app_seq.append(app)
                time_seq.append(time)
                location_seq.append(location)

        prev_user = user

    pri_dataset = pd.DataFrame()
    pri_dataset['user'] = all_user_seq
    pri_dataset['time_seq'] = all_time_seq
    pri_dataset['app_seq'] = all_app_seq
    pri_dataset['location_vectors_seq'] = all_location_seq
    pri_dataset.to_csv(os.path.join(path, name), sep='\t', index=False)
    return pri_dataset

pri_len = 5
idx = 'test_1'

if not os.path.exists('data/Tsinghua_new//time_division/' + idx + '/primary/'):
    os.makedirs('data/Tsinghua_new//time_division/' + idx + '/primary/')

df_usage = pd.read_csv('data/Tsinghua_new/App_usage_trace.txt', sep=' ',
                       names=['user', 'date', 'location', 'app', 'traffic'])

date1 = df_usage['date']
df_usage['date1'] = date1

df_usage['date1'] = df_usage['date1'].apply(lambda x: str(x)[:-6])
df_usage['date'] = df_usage['date'].apply(lambda x: str(x)[:-2])

df_usage.drop_duplicates(subset=['user', 'date', 'app'], inplace=True)

# delete apps used less than 10 times for all users
df_usage = df_usage[df_usage.groupby('app')['app'].transform('count').ge(10)]

# 添加location的编码

# 读取数据
data = pd.read_csv('data/Tsinghua_new/location_clustering/base_poi.txt', delimiter='\t', header=0)
labels = data.iloc[:, 0]
vectors = data.iloc[:, 1:]

# 数据预处理
scaler = StandardScaler()
scaled_vectors = scaler.fit_transform(vectors)

# 使用K-Modes聚类算法对向量进行聚类
kmodes = KModes(n_clusters=5, random_state=0).fit(scaled_vectors)

# 输出每个标签对应的类别
output_dict = {}
for i, label in enumerate(labels):
    output_dict[label] = kmodes.cluster_centroids_[kmodes.labels_[i]].tolist()

# 保存结果到txt文件
with open('data/Tsinghua_new/location_clustering/output.txt', 'w') as f:
    for label, values in output_dict.items():
        f.write(f'{label}\t{values}\n')
    f.write('\n')

# 保存结果到pickle文件
with open('data/Tsinghua_new/location_clustering/output.pickle', 'wb') as f:
    pickle.dump(output_dict, f)

# 加载pickle文件
with open('data/Tsinghua_new/location_clustering/output.pickle', 'rb') as f:
    output_dict = pickle.load(f)

# 将原始数据集中的标签列替换为其所属类别对应的类中心向量
df_usage['location_vectors'] = df_usage['location'].map(output_dict)

# 将主网络训练集中的App根据主网络App编号表重新编号，未在编号表中出现的App编号为0
pri_app2id = generate_pri_app2id(df_usage, idx=idx)
pri_app = pri_app2id.keys()
df_usage['app'] = df_usage['app'].apply(lambda x: pri_app2id[x] if x in pri_app else 0)

if not os.path.exists('data/Tsinghua_new/time_division/division/'):
    os.makedirs('data/Tsinghua_new/time_division/division/')

# 根据日期（日）进行分组
for name, group in df_usage.groupby('date1'):
    num_a = group['date1'].count() # 获取组内记录数目
    print(name)
    print(num_a)
    # 去掉date1这一列
    group = group.drop('date1', axis=1)

    group.to_csv('data/Tsinghua_new/time_division/division/' + name + '.txt', sep=' ', index=False)
    print('------------------------')

# 拼接前五天的数据
data1 = pd.read_csv('data/Tsinghua_new/time_division/division/20160420.txt', sep=' ', header=0)  # data1为DataFrame格式
data2 = pd.read_csv('data/Tsinghua_new/time_division/division/20160421.txt', sep=' ', header=0)
data3 = pd.read_csv('data/Tsinghua_new/time_division/division/20160422.txt', sep=' ', header=0)
data4 = pd.read_csv('data/Tsinghua_new/time_division/division/20160423.txt', sep=' ', header=0)
data5 = pd.read_csv('data/Tsinghua_new/time_division/division/20160424.txt', sep=' ', header=0)

data1.to_csv('data/Tsinghua_new/time_division/division/five_data.txt', sep=' ', index=False, header=False)
data2.to_csv('data/Tsinghua_new/time_division/division/five_data.txt', sep=' ', index=False, header=False, mode='a+')
data3.to_csv('data/Tsinghua_new/time_division/division/five_data.txt', sep=' ', index=False, header=False, mode='a+')
data4.to_csv('data/Tsinghua_new/time_division/division/five_data.txt', sep=' ', index=False, header=False, mode='a+')
data5.to_csv('data/Tsinghua_new/time_division/division/five_data.txt', sep=' ', index=False, header=False, mode='a+')

five = pd.read_csv('data/Tsinghua_new/time_division/division/five_data.txt', sep=' ', names=['user', 'date', 'location', 'app', 'traffic', 'location_vectors'])

if not os.path.exists('data/Tsinghua_new/time_division/division/five_user/'):
    os.makedirs('data/Tsinghua_new/time_division/division/five_user/')

# 再根据用户进行分组
for name, group in five.groupby('user'):
    name = str(name)
    group.to_csv('data/Tsinghua_new/time_division/division/five_user/' + name + '.txt', sep=' ', index=False)
    data = pd.read_csv('data/Tsinghua_new/time_division/division/five_user/' + name + '.txt', sep=' ', header=0)
    data.to_csv('data/Tsinghua_new/time_division/division/five.txt', sep=' ', index=False, header=False, mode='a+')

print('-----------five-------------')

# 读取前五天的数据
five1 = pd.read_csv('data/Tsinghua_new/time_division/division/five.txt', sep=' ', names=['user', 'date', 'location', 'app', 'traffic', 'location_vectors'])
num_b = five1.shape[0]
print('five1：', num_b)
print('------------------------')

# five1 = five1.astype({'date': 'string'})
five1['date'] = five1['date'].astype(str)

# 生成第七天比前五天多的序号
# 生成按five.txt包含的app序号
five_app = pd.read_csv('data/Tsinghua_new/time_division/division/five.txt', sep=' ', names=['user', 'date', 'location', 'app', 'traffic', 'location_vectors'])
# 删掉重复的user编号
five_app.drop_duplicates(subset=['app'], inplace=True)
five_app1 = five_app['app']

five_app1 = five_app1.values.tolist()

if not os.path.exists('data/Tsinghua_new/time_division/app_serial_number/'):
    os.makedirs('data/Tsinghua_new/time_division/app_serial_number/')

file = open('data/Tsinghua_new/time_division/app_serial_number/five_app.txt', 'w')
file.write(str(five_app1))
file.close()

# 生成20160426.txt包含的app序号
seventh_app = pd.read_csv('data/Tsinghua_new/time_division/division/20160426.txt', sep=' ')
# 删掉重复的user编号
seventh_app.drop_duplicates(subset=['app'], inplace=True)
seventh_app1 = seventh_app['app']

seventh_app1 = seventh_app1.values.tolist()
file = open('data/Tsinghua_new/time_division/app_serial_number/seventh_app.txt', 'w')
file.write(str(seventh_app1))
file.close()

# 排序
five_app1.sort()
seventh_app1.sort()

# 列表相减
a = list(set(seventh_app1) - set(five_app1))
b = list(set(five_app1) - set(seventh_app1))

file = open('data/Tsinghua_new/time_division/app_serial_number/seventh-five.txt', 'w')
file.write(str(a))
file.close()

file = open('data/Tsinghua_new/time_division/app_serial_number/five-seventh.txt', 'w')
file.write(str(b))
# 结束读文件
file.close()

# 读取第七天的数据
# 将第七天多出前五天的app序号设为0
a = dict.fromkeys(a)
a = a.keys()
seventh1 = pd.read_csv('data/Tsinghua_new/time_division/division/20160426.txt', sep=' ')
seventh1['app'] = seventh1['app'].apply(lambda x: 0 if x in a else x)

num_d = seventh1.shape[0]
print('seventh1：', num_d)
print('------------------------')
seventh1['date'] = seventh1['date'].astype(str)

# 根据需要的数据输入长度（pri_len），将数据集切分成多个子序列，也就是数据输入样本
train_pri = generate_primary_dataset(five1, path='data/Tsinghua_new/time_division/' + idx + '/primary/', length=pri_len, name='train.txt')

test_pri = generate_primary_dataset(seventh1, path='data/Tsinghua_new/time_division/' + idx + '/primary/', length=pri_len, name='test.txt')




