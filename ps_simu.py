#!/usr/bin/env python
# coding: utf-8

# データ系
import pandas as pd
import random
import numpy as np
from collections import Counter

# 作図系
import matplotlib.pyplot as plt
import japanize_matplotlib

# csv読み込み
df = pd.read_csv('sleepStyleDictionary.csv')
howManyMonsAppear = pd.read_csv('howManyMonsAppear.csv')

# 計算精度
sample = 100000 #投稿時は10000

# oupputの有無
output = False #Trueのときpngとfigを生成

# 寝顔リサーチでのパラメータ
sleepScore = 100 #GSDのとき200相当
fields = ['ワカクサ', 'シアン', 'トープ', 'ウノハナ']
sleepTypes = ['うとうと', 'すやすや', 'ぐっすり']
rewards = ['SPO','経験値アメ','リサーチEXP','ゆめのかけら']
nums = ['出現数','経験値アメ']

# カラーとラインスタイルの設定
color = {'ワカクサ': 'yellowgreen',
         'シアン': 'aqua',
         'トープ': 'brown',
         'ウノハナ': 'steelblue'}

line = {'うとうと': ':',
        'すやすや': '--',
        'ぐっすり': '-'}

marker = {'うとうと': '^',
          'すやすや': 'o',
          'ぐっすり': 's'}


# 主要な関数(1,2 -> 3 の順番で使う)
# 1. 出現寝顔数の決定
def howMany(field,drowsyPower):
    condition  = howManyMonsAppear['フィールド'] == field
    condition &= howManyMonsAppear['必要ねむけパワー'] <= drowsyPower
    temp = howManyMonsAppear[condition]
    return temp['寝顔数'].max()

# 2. 抽選テーブルの辞書を作成
def filteredDict(field,sleepType,drowsyPower):
    condition  = (df['フィールド'] == field)
    condition &= (df['睡眠タイプ'] == sleepType)
    condition &= (df['エナジー'] < drowsyPower/sleepScore)#未解放の寝顔を除外
    condition &= (df['ポケモン'].apply(len) < 6) #ホリデー&ハロウィン除外用

    df_temp = df[condition]

    dictN = df_temp.set_index(['ポケモン', '星']).to_dict(orient='index')

    #絶食厳選
    df_spo2 = df_temp[(df_temp['SPO']==2)&(df_temp['エナジー']==0)]
    df_0Nrgy = df_spo2.loc[df_spo2['寝顔ID'].idxmin()]

    key_dict0 = (df_0Nrgy['ポケモン'],df_0Nrgy['星'])

    return dictN,key_dict0


# 3. 寝顔抽選
def selectedRows(dp,n):

    rows = []
    spo_remain = dp / 38000
    on_berry = False

    while len(rows) < n:
        if spo_remain < 2:

            rows.append(key_dict0)
            spo_remain -= dictN[key_dict0]['SPO']

        else:
            if len(rows) < n - 1:

                dict_random = random.choice(list(dictN))

                if dictN[dict_random]['SPO'] > spo_remain:
                    continue

                if dict_random[1] == '4':
                    if on_berry:
                        continue
                    else:
                        on_berry = True

                rows.append(dict_random)
                spo_remain -= dictN[dict_random]['SPO']

            else:

                max_spo_element = max(
                    ((key, value) for key, value in dictN.items() if value['SPO'] <= spo_remain),
                    key=lambda x: x[1]['SPO']
                )

                rows.append(max_spo_element[0])
                spo_remain -= max_spo_element[1]['SPO']

    return rows


# 結果1. 報酬期待値
# 計算パート
y_reward = {}
e_reward = {}
x = np.linspace(3e6,4e8,30) #np.logspace(3,8,20)
x_str = [i/1e8 for i in x]

for field in fields:
    for sleepType in sleepTypes:
        for reward in rewards:
            y_reward[(field,sleepType,reward)]=[]
            e_reward[(field,sleepType,reward)]=[]

        for drowsyPower in x:
            dictN,key_dict0 = filteredDict(field,sleepType,drowsyPower)
            n = howMany(field,drowsyPower)

            sum_reward = {}
            for reward in rewards:
                sum_reward[reward] = []

            for s in range(sample):
                rows = selectedRows(drowsyPower,n)
                for reward in rewards:
                    sum_reward[reward].append(sum(dictN[row][reward] for row in rows))

            for reward in rewards:
                y_reward[(field,sleepType,reward)].append(np.mean(sum_reward[reward]))
                e_reward[(field,sleepType,reward)].append(np.std(sum_reward[reward])/np.sqrt(sample))


# 作図パート
for reward in rewards:
    plt.figure(figsize=(6, 4))
    for field in fields:
        for sleepType in sleepTypes:
            plt.plot(x_str,
                     y_reward[(field,sleepType,reward)],
                     label=field +'-'+ sleepType,
                     marker = marker[sleepType],
                     color=color[field],
                     linestyle=line[sleepType])

    plt.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
    plt.xlabel('眠気パワー（単位：億）')
    plt.ylabel(reward)
    plt.grid(axis='y')
    #plt.xscale('log')
    if output: plt.savefig(reward + '.png',bbox_inches='tight')
    plt.show()


# 結果2. 個別の出現数とアメ数
targets = ['イワーク','デリバード','アブソル','メタモン','ヒノアラシ','チコリータ','ワニノコ','タマザラシ','ピィ','ゼニガメ','ゴース','ディグダ','マダツボミ','イーブイ','ププリン']

# 計算パート
y_target = {}
e_target = {}
x = np.linspace(3e6,3e8,50)
x_str = [i/1e8 for i in x]

for field in fields:
    for sleepType in sleepTypes:
        for target in targets:
            for num in nums:
                y_target[(field,sleepType,target,num)]=[]
                e_target[(field,sleepType,target,num)]=[]

        for drowsyPower in x:
            dictN,key_dict0 = filteredDict(field,sleepType,drowsyPower)
            n = howMany(field,drowsyPower)

            sum_target = {}
            for target in targets:
                for num in nums:
                    sum_target[(target,num)] = []

            for s in range(sample):
                rows = selectedRows(drowsyPower,n)
                for target in targets:
                    sum_target[(target,'出現数')].append(sum(row[0] == target for row in rows))
                    sum_target[(target,'経験値アメ')].append(sum(dictN[row]['経験値アメ'] for row in rows if dictN[row]['種ポケモン'] == target))

            for target in targets:
                for num in nums:
                    y_target[(field,sleepType,target,num)].append(np.mean(sum_target[(target,num)]))
                    e_target[(field,sleepType,target,num)].append(np.std(sum_target[(target,num)])/np.sqrt(sample))


# 作図パート
for target in targets:
    plt.figure(figsize=(4, 3))
    for field in fields:
        for sleepType in sleepTypes:
            if sum(y_target[(field,sleepType,target,'経験値アメ')])==0:continue
            for num in nums:
                plt.subplot(2,1,nums.index(num)+1)
                plt.errorbar(x_str,
                             y_target[(field,sleepType,target,num)],
                             e_target[(field,sleepType,target,num)],
                             linestyle = line[sleepType],
                             label= field +'-'+ sleepType,
                             color=color[field])

    for num in nums:
        plt.subplot(2,1,nums.index(num)+1)
        plt.ylabel(num)
        plt.ylim(bottom=0)
        plt.grid()
        if num == '出現数':
            plt.title(target)
            plt.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
        else:
            #plt.legend(fontsize=8)
            plt.xlabel('眠気パワー（単位：億）')
    if output: plt.savefig(target + '.png',bbox_inches='tight')
    plt.show()


# 結果3. すべての出現数とアメ数
# カウンター
def counter(data,y):
    counter = Counter(data)
    labels = list(counter.keys())
    counts = list(counter.values())
    for mon in y:
        if mon in labels:
            y[mon].append(counts[labels.index(mon)]/sample)
        else:
            y[mon].append(0)
    return y

# 帯グラフでの色
pkmn_type_colors = {
'Grass':'#2f9958',
'Fire':'#FF5020',
'Water':'#74d7d7',
'Bug':'#A8B820',
'Normal':'#BBAAAA',
'Dark':'#111111',
'Steel':'#b5bed7',
'Flying':'#8198FF',
'Poison':'#9e69be',
'Electric':'#F8E830',
'Ground':'#E0C068',
'Fairy':'#FEB9CC',
'Fighting':'#F08068',
'Psychic':'#D85888',
'Rock':'#A89038',
'Ghost':'#525364',
'Ice':'#B8E8D8',
'Dragon':'#7038F8'}

def hex_noise(hex_color):
    # 16進数表現からRGBに変換
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0

    # ノイズを付与
    noise_factor = 0.1
    noise = np.random.uniform(low=-noise_factor, high=noise_factor, size=3)
    r,g,b = np.clip(np.array((r,g,b)) + noise, 0.0, 1.0)

    return r, g, b

# ポケモン固有のカラーを設定
color_type = pd.read_csv('type.csv')
colors = dict(zip(color_type['名前'],
                  [hex_noise(pkmn_type_colors[t]) for t in color_type['Type']]))

# 計算パートと作画パート
x = [1e5,3e5,1e6,3e6,1e7,3e7,1e8,3e8,1e9]
x_str = [str(i/1e8) for i in x]

y_mon_output = {} # 出現数
y_candy_output = {} # 経験値アメ数

for field in fields:
    for sleepType in sleepTypes:

        y_mon = {}
        y_candy = {}
        dictN,key_dict0 = filteredDict(field,sleepType,1e10)

        for key,value in dictN.items():
            y_mon[key[0]] = []
            y_candy[dictN[key]['種ポケモン']] = []

        for drowsyPower in x:
            dictN,key_dict0 = filteredDict(field,sleepType,drowsyPower)
            n = howMany(field,drowsyPower)

            mon = []
            candy = []
            for s in range(sample):
                rows = selectedRows(drowsyPower,n)
                for row in rows:
                    mon.append(row[0])
                    for i in range(dictN[row]['経験値アメ']):
                        candy.append(dictN[row]['種ポケモン'])


            y_mon = counter(mon,y_mon)
            y_candy = counter(candy,y_candy)

        #output用
        for mon in y_mon:
            y_mon_output[(mon,field,sleepType)]=y_mon[mon]
        for mon in y_candy:
            y_candy_output[(mon,field,sleepType)]=y_candy[mon]

        #グラフの積み上げ
        bottom = np.zeros(len(x_str))
        num = 0
        fig = plt.figure(figsize=(10, 6))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for num in nums:
            plt.subplot(1,2,nums.index(num)+1)
            plt.xlabel('眠気パワー(単位：億)')
            plt.xticks(rotation='vertical')
            plt.ylabel(num)
            plt.title(field + '-' + sleepType,fontsize=10)

            bottom = np.zeros(len(x_str))
            pileup = 0

            if num == '出現数':
                y = y_mon

            else:
                y = y_candy

            for mon in dict(reversed(y.items())):
                # 積み上げ棒グラフを描く
                plt.bar(x_str, y[mon],bottom=bottom,label=mon,color=colors[mon],edgecolor='white',linewidth=0.5)
                #,color=color_mon[mon])
                bottom += y[mon]
                pileup += 1

            plt.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8,reverse=True)
            #plt.grid(axis='y')

        if output: plt.savefig(field + '-' + sleepType + '.png',bbox_inches='tight')
        plt.show()

# CSVアウトプット用
df_mon_output = pd.DataFrame(y_mon_output)
df_mon_output.index = x_str

df_candy_output = pd.DataFrame(y_candy_output)
df_candy_output.index = x_str
df_candy_output
if output: df_candy_output.to_csv('num_candy_research.csv')

if output:
    df_mon_output.transpose().to_csv('num_mon_research.csv', header=True)
    df_candy_output.transpose().to_csv('num_candy_research_.csv', header=True)
    df_mon_output.transpose().to_csv('num_mon_research_shift-jis.csv', sep='|', header=True, encoding='shift_jis', float_format='%.2f')
    df_candy_output.transpose().to_csv('num_candy_research_shift-jis.csv', sep='|', header=True, encoding='shift_jis', float_format='%.2f')