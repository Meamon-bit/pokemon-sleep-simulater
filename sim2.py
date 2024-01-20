#!/usr/bin/env python
# coding: utf-8

# データ系
import pandas as pd
from flask import Flask, render_template, request
import random
import numpy as np
from collections import Counter

# csv読み込み
df = pd.read_csv('sleepStyleDictionary.csv')
howManyMonsAppear = pd.read_csv('howManyMonsAppear.csv')

# 寝顔リサーチでのパラメータ
sleepScore = 100 #GSDのとき200相当
fields = ['ワカクサ', 'シアン', 'トープ', 'ウノハナ']
sleepTypes = ['うとうと', 'すやすや', 'ぐっすり']
rewards = ['SPO','経験値アメ','リサーチEXP','ゆめのかけら']
nums = ['出現数','経験値アメ']

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
    condition &= (df['ポケモン'].apply(len) < 7) #ホリデー&ハロウィン除外用

    df_temp = df[condition]

    dictN = df_temp.set_index(['ポケモン', '星']).to_dict(orient='index')

    #絶食厳選
    df_spo2 = df_temp[(df_temp['SPO']==2)&(df_temp['エナジー']==0)]
    df_0Nrgy = df_spo2.loc[df_spo2['寝顔ID'].idxmin()]

    key_dict0 = (df_0Nrgy['ポケモン'],df_0Nrgy['星'])

    return dictN,key_dict0


# 3. 寝顔抽選
def selectedRows(dp,n,key_dict0,dictN):

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

                if dict_random[1] == 4:
                    if on_berry:
                        continue
                    else:
                        on_berry = True

                rows.append(dict_random)
                spo_remain -= dictN[dict_random]['SPO']

            else:


                if on_berry:
                    temp = [(key, value['SPO']) for key, value in dictN.items()
                            if value['SPO'] <= spo_remain and key[1] != 4]
                else:
                    temp = [(key, value['SPO']) for key, value in dictN.items()
                            if value['SPO'] <= spo_remain]

                max_spo = max(value for key,value in temp)
                elements_max_spo = [key for key,value in temp if value == max_spo]
                key_final = random.choice(elements_max_spo)

                rows.append(key_final)
                spo_remain -= max_spo

    return rows

# 0. Web上での入出力処理
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        # フォームからデータを取得
        field = request.form.get('researchfield')
        sleepType = request.form.get('sleepType')
        energy = request.form.get('energy')
        sleepScore = 100
        drowsyPower = int(energy)*sleepScore
        n = howMany(field,drowsyPower)
        dictN,key_dict0 = filteredDict(field,sleepType,drowsyPower)
        result = selectedRows(drowsyPower,n,key_dict0,dictN)
        # print(result)
        sum_reward = {}
        for reward in rewards:
            sum_reward[reward] = []
        for reward in rewards:
            sum_reward[reward].append(sum(dictN[row][reward] for row in result))
        result.append(sum_reward)
        # print(sum_reward)
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

