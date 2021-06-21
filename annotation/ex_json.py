list1 = ['122', '2333', '3444', '', '', None]
a = list(filter(None, list1))  # 只能过滤空字符和None
print(a)  # ['122', '2333', '3444']


# Python内建filter()函数 - 过滤list
# filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素
def not_empty(s):
    return s and s.strip()


list2 = ['122', '2333', '3444', ' ', '422', ' ', '    ', '54', ' ', '', None, '   ']
print(list(filter(not_empty, list2)))  # ['122', '2333', '3444', '422', '54']
# 不仅可以过滤空字符和None而且可以过滤含有空格的字符