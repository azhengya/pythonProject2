# -*- coding:utf-8 -*-
import xlrd
# "authors": [
#
# { "firstName": "Isaac", "lastName": "Asimov", "genre": "science fiction" },
#
# { "firstName": "Tad", "lastName": "Williams", "genre": "fantasy" },
#
# { "firstName": "Frank", "lastName": "Peretti", "genre": "christian fiction" }
#
# ],
def read_xls(filename):
    # 打开Excel文件,读取第一个工作表
    data = xlrd.open_workbook(filename).sheets()[0]
    # data.nrows统计行数，ncols为列数
    cols = data.ncols
    # 保存结果
    result = []
    #for i in range(cols):
    for j in data.col_values(1):
        print(data.col_values(1)[1])
        con = 0
        if j != '':
            con +=1
        print(con)

    #[print(len([x for x in w if x != ''])) for w in data.row(1)]
    # l = len(filter(None, data.row(1)))
    # print(l)
    for i in range(cols):
        record = {}
        record[str(int(data.col_values(i)[0]))]= data.col_values(i)[1:]
        result.append(record)
    return result
            # record[keys[0]] = cols.value[0]
            # keys = data.col_values(i)  # 保存关键字
        # else:
        #     record = {}
        #     record[keys[0]] = data.col_values(0)
        #     # 将Excel文件的数据存入字典中
        #     cnt = 1
        #     for item in data.col_values(i):
        #         record[keys[cnt]] = item
        #         cnt += 1
        #     # 将字典存入列表
        #     result.append(record)
# 去空

def not_empty(s):
    return s and s.strip()

if __name__ == '__main__':

    d1 = str(read_xls("./测试.xlsx"))
        # for key, value in i.items():
        #     print(key, value)
    # for i in range(len(d1)):
    #print([item[key] for item in d1 for key in item] )
    print(d1)
    # 可读可写，如果不存在则创建，如果有内容则覆盖
    jsFile = open("./11.xml", "w+", encoding='utf-8')
    jsFile.write(d1)
    jsFile.close()
