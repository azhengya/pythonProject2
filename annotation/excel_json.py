#coding=gbk
import xlrd

def read_xls(filename):

    # ��Excel�ļ�
    data = xlrd.open_workbook(filename)

    # ��ȡ��һ��������
    table = data.sheets()[0]

    # ͳ������
    rows = table.nrows

    data = []   # �������
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

    d1 = read_xls("./ģ����-2.xlsx")
    d2 = str(d1).replace("\'", "\"")    # �ֵ��е����ݶ��ǵ����ţ����Ǳ�׼��json��Ҫ˫����

    # d3 = str(d2).replace("''","")
    print(d2)

    d2 = "{\"DeviceList\":" + d2 + "}"    # ǰ�������ֻ�����飬���������json��ʽ������

    # �ɶ���д������������򴴽�������������򸲸�
    jsFile = open("./DevicesInfo.js", "w+", encoding='utf-8')
    jsFile.write(d2)
    jsFile.close()
