# -*- coding:utf-8 -*-
import xlrd

def read_xls(filename):
    # 打开Excel文件,读取第一个工作表
    data = xlrd.open_workbook(filename).sheets()[2]
    # data.nrows统计行数，ncols为列数
    rows = data.ncols
    # 保存结果
    dc = []

    for i in range (rows):
        ls = []
        for j in data.col_values(i):
            j = str(j)
            # print(data.col_values(i))
            if j != '':
                ls.append(j)

        dc.extend(ls)
    print(dc)
    print(len(dc))
    # 遍历字典列表
    # for key, values in dc.items():
    #     print (key, values)
    return dc


if __name__ == '__main__':

    d1 = str(read_xls("/home/dd-29/桌面/文件文档/转.xlsx"))
    # print(d1)
    # 可读可写，如果不存在则创建，如果有内容则覆盖
    jsFile = open("./excel_json.xml", "w+", encoding='utf-8')
    jsFile.write(d1)
    jsFile.close()
