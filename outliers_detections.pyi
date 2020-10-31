import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
'''
    이상점 검출기법
    - 백분위수 기반의 이상점 검출
    - 중위 절대 편차 기반의 이상점 검출
    - 표준편차 기반의 이상점 검출
    - 다수결 투표기반의 이상점 검출
    - 이상점의 시각화 
'''
# 백분위수 기반의 이상점 검출 threshold: 분계점
'''
    백분위수 기반: 기본 통계적 이해를 바탕으로 파생된 백분위수 기반의 이상 검출
'''


def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    (minval, maxval) = np.percentile(data, [diff, 100 - diff])
    return ((data < minval) | (data > maxval))


# 중위 절대 편차 기반 이상점 검출
'''
    중위 절대 편차: 간단한 통계 개념
    이 개념에는 4단계가 관계되어있음
    1. 특정 데이터 특성의 중위수(median)를 찾는다
    2. 데이터 특성에 대해 주어진 각 값에서 이전에 찾은 중위수를 뺀다.
       이 뺄셈은 절댓값 형식이다 따라서 각 데이터 점(data point)에 대해서 절대값을 얻게 될것
    3. 세 번째 단계에서는 두 번째 단계에서 얻은 절대값의 중위수를 생성한다. 각 데이터 특성에 대한 각 데이터 점에대해 이 작업을
       수행한다. 이 값을 MAD값이라 한다.
    4. 네번째 단게에서는 수정 z 점수(중위 절대 편차)를 유도하기위해 (0.6475 * (xi-x~)/mad)를 사용할것 MAD=median절대값 Yi-Y~
    Y~는 해당데이터의 중위수
'''


def mad_based_outlier(points, threshold=3.5):
    if len(points.shape) == 1:
        points = points[:None]
    median_y = np.median(points)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in points])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in points]
    return np.abs(modified_z_scores) > threshold


# 표준편차 기반 이상점 검출
'''
    표준편차와 평균 값을 사용해 이상점을 찾음. 여기서는 임의의 분계점인 3을 선택
'''


def std_div(data, threshold=3):
    std = data.std()
    mean = data.mean()
    isOutlier = []
    for val in data:
        if val / std > threshold:
            isOutlier.append(True)
        else:
            isOutlier.append(False)
    return isOutlier


'''
    다수결 투표 기반 이상점 검출:
    투표 방식을 구축해 백분위 기반 이상점 검출 ,MAD 기반 이상 검출, std 기반 이상점 검출과 같은 이전에 정의된 모든방법을 동시에 실행함으로써 데이터점을 이상점으로 봐야할 지
    이상점이 아니라고 봐야할지를 알수 있게 한다.

'''


def outlierVote(data):
    x = percentile_based_outlier(data)
    y = mad_based_outlier(data)
    z = std_div(data)

    temp = list(zip(data.index, x, y, z))
    final = []
    for i in range(len(temp)):
        if temp[i].count(False) >= 2:
            final.append(False)
        else:
            final.append(True)
    return final


# 이상점 시각화

def plotOutlier(x):
    fig, axes = plt.subplots(nrows=4)
    for ax, func in zip(axes, [percentile_based_outlier, mad_based_outlier, std_div, outlierVote]):
        sns.distplot(x, ax=ax, rug=True, hist=False)
        outliers = x[func(x)]
        ax.plot(outliers, np.zeros_like(outliers), 'ro', clip_on=False)

    kwargs = dict(y=0.95, x=0.05, ha='left', va='top', size=20)
    axes[0].set_title('Percentile-based outliers', **kwargs)
    axes[1].set_title('MAD-based outliers', **kwargs)
    axes[2].set_title('STD-based outliers', **kwargs)
    axes[3].set_title('majority vpte based', **kwargs)
    fig.suptitle('Comparing Outlier Test with n={}'.format(len(x)), size=20)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.show()