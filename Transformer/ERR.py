def cross_point(line1, line2):
    # 取直线坐标两点的x和y值
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    # L2直线斜率不存在操作
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
        x = x3
        # 计算k1,由于点均为整数，需要进行浮点数转化
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        # 整型转浮点型是关键
        b1 = y1 * 1.0 - x1 * k1 * 1.0
        y = k1 * x * 1.0 + b1 * 1.0
    elif (x2 - x1) == 0:
        k1 = None
        b1 = 0
        x = x1
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        y = k2 * x * 1.0 + b2 * 1.0
    else:
        # 计算k1,由于点均为整数，需要进行浮点数转化
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        # 斜率存在操作
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        # 整型转浮点型是关键
        b1 = y1 * 1.0 - x1 * k1 * 1.0
        b2 = y3 * 1.0 - x3 * k2 * 1.0
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]
line1=[1.4,      3.81578947 ,1.5,4.86842105    ]
line2=[1.4,   5.25,1.5, 4.25]
# print(cross_point(line1,line2))