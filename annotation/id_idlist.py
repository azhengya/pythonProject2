# -*- coding:utf-8 -*-
import xlrd

def read_xls(filename):
    # 打开Excel文件,读取第一个工作表
    data = xlrd.open_workbook(filename).sheets()[0]
    # data.nrows统计行数，ncols为列数
    cols = data.ncols
    # 保存结果
    dc = []

    for i in range (cols):
        ls = []
        for j in data.col_values(i):
            j = int(j)
            # print(data.col_values(i))
            if j != '':
                ls.append(j)
        dc.extend(ls)
    print(dc)
    return dc


if __name__ == '__main__':

    d1 = str(read_xls("类iid.xls"))
    # print(d1)
    # 可读可写，如果不存在则创建，如果有内容则覆盖
    jsFile = open("./excel_json.xml", "w+", encoding='utf-8')
    jsFile.write(d1)
    jsFile.close()
