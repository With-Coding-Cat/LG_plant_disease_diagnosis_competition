csv_columns = ['측정시각','내부 온도 1 평균','내부 온도 1 최고','내부 온도 1 최저','내부 온도 2 평균','내부 온도 2 최고',
               '내부 온도 2 최저','내부 온도 3 평균','내부 온도 3 최고','내부 온도 3 최저','내부 온도 4 평균','내부 온도 4 최고',
               '내부 온도 4 최저','내부 습도 1 평균','내부 습도 1 최고','내부 습도 1 최저','내부 습도 2 평균','내부 습도 2 최고',
               '내부 습도 2 최저','내부 습도 3 평균','내부 습도 3 최고','내부 습도 3 최저','내부 습도 4 평균','내부 습도 4 최고',
               '내부 습도 4 최저','내부 이슬점 평균','내부 이슬점 최고','내부 이슬점 최저','내부 CO2 평균','내부 CO2 최고',
               '내부 CO2 최저','외부 풍속 평균','외부 풍속 최고','외부 풍속 최저','내부 EC 1 평균','내부 EC 1 최고','내부 EC 1 최저',
               '내부 PH 1 평균','내부 PH 1 최고','내부 PH 1 최저','배지 중량 평균','배지 중량 최고','배지 중량 최저','양액 온도 평균',
               '양액 온도 최고','양액 온도 최저','외부 풍향 수치','외부 풍향','외부 빗물 시간','외부 누적일사 평균','양액 급액 누적',
               '양액 배액 누적']

use_columns = ['내부 온도 1 평균','내부 온도 1 최고','내부 온도 1 최저',
               '내부 습도 1 평균','내부 습도 1 최고','내부 습도 1 최저',
               '내부 이슬점 평균','내부 이슬점 최고','내부 이슬점 최저',
               '내부 CO2 평균','내부 CO2 최고','내부 CO2 최저',
               '외부 누적일사 평균']

disease_dict = {0:['0'],
                1:['00', 'a5'],
                2:['00', 'a9', 'b3', 'b6', 'b7', 'b8'],
                3:['0'],
                4:['00', 'a7', 'b6', 'b7', 'b8'],
                5:['00', 'a11', 'a12', 'b4', 'b5'],}
real_targets = ['1_00_0', '2_00_0', '2_a5_2', '3_00_0', '3_a9_1', '3_a9_2', '3_a9_3', '3_b3_1', '3_b6_1', '3_b7_1', '3_b8_1', '4_00_0', '5_00_0', '5_a7_2', '5_b6_1', '5_b7_1', '5_b8_1', '6_00_0', '6_a11_1', '6_a11_2', '6_a12_1', '6_a12_2', '6_b4_1', '6_b4_3', '6_b5_1']

disease_encoding = []
for disease_codes in disease_dict.values():
    disease_encoding.extend(disease_codes)
disease_encoding = sorted(list(set(disease_encoding)))
disease_decoding = {}
for k, v in enumerate(disease_encoding):
    disease_decoding[k] = v

disease_mask = {}
for k, values in disease_dict.items():
    disease_mask[k] = [True] * len(disease_encoding)
    for v in values:
        disease_mask[k][disease_encoding.index(v)] = False

percentile_num = 50

risk_with_crop_mask = {}
for target_name in real_targets:
    crop_i, disease_i, risk_i = target_name.split('_')
    crop_i = int(crop_i) - 1
    if crop_i == 0 or crop_i == 3:
        disease_i = disease_encoding.index('0')
    else:
        disease_i = disease_encoding.index(disease_i)
    risk_i = int(risk_i)
    risk_with_crop_mask[disease_i*10 + crop_i] = [True, True, True, True]
for target_name in real_targets:
    crop_i, disease_i, risk_i = target_name.split('_')
    crop_i = int(crop_i) - 1
    if crop_i == 0 or crop_i == 3:
        disease_i = disease_encoding.index('0')
    else:
        disease_i = disease_encoding.index(disease_i)
    risk_i = int(risk_i)
    risk_with_crop_mask[disease_i*10 + crop_i][risk_i] = False