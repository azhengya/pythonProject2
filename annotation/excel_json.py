#coding=gbk
import xlrd

def read_xls(filename):

    # 打开Excel文件
    data = xlrd.open_workbook(filename)

    # 读取第一个工作表
    table = data.sheets()[0]

    # 统计行数
    rows = table.nrows

    data = []   # 存放数据
    for v in range(1, rows):
        values = table.row_values(v)
        data.append(
            (
                {
                str(values[0]): str(values[1:])
                }
            )
     )


    return data



if __name__ == '__main__':

    d1 = read_xls("./模型类-2.xlsx")
    d2 = str(d1).replace("\'", "\"")    # 字典中的数据都是单引号，但是标准的json需要双引号

    # d3 = str(d2).replace("''","")
    print(d2)

    d2 = "{\"DeviceList\":" + d2 + "}"    # 前面的数据只是数组，加上外面的json格式大括号

    # 可读可写，如果不存在则创建，如果有内容则覆盖
    jsFile = open("./DevicesInfo.js", "w+", encoding='utf-8')
    jsFile.write(d2)
    jsFile.close()
