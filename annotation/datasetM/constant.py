# _*_coding:utf-8_*_
# __author: guo
merge_data = {
    "dddp": {
        "rename": {
            "鲜枣": "鲜红枣",
            "红枣": "红枣核",
            "塑料袋装牛奶": "酸奶袋",
            "生煎包子": "生煎",
            "水饺": "速冻水饺",
        },
        "merge": {"打汁机": ['榨汁机', '豆浆机'], '室内垃圾桶': ['室内垃圾桶', '闭合垃圾桶'], '床单被套': ['床单被套', '毛毯'],
                  '纸巾包装袋': ['抽纸塑料袋', '纸巾包装袋'], '鸡蛋': ['鸡蛋', '蛋壳', '茶叶蛋'], '电风扇': ['手持电风扇', '电风扇'],
                  '豆腐': ['豆腐', '冻豆腐'], '衣服': ['衣服', '牛仔裤'], '常见肉类': ['牛肉牛排', '烤肉', '鸡胸肉', '熟里脊肉', '常见肉类'],
                  '火腿肠': ['烤肠', '火腿肠'], '笔': ['眼线笔', '白板笔', '蜡笔', '油漆笔', '木头铅笔', '荧光笔', '笔'],
                  '生菜包菜': ['包菜', '生菜', '生菜包菜'], '书纸杂志': ['书纸', '纸', '杂志', '测试卷子', '报纸', '书纸杂志'],
                  '鞋': ['皮鞋', '一次性拖鞋', '鞋'], '滑冰鞋': ['双排旱冰鞋', '单排旱冰鞋', '滑冰鞋']},
        "remove": [
            "笔盒", "水果果皮", "剩菜剩饭",
            "垫子", "饺子皮", "鸡蛋皮", "包装用纸", "del", "unknown",
            "吉他鼓", "钢琴", "足球", "橱柜", "门", '洗碗机',
            '长笛', '奶茶中的珍珠', '手风琴', "灯罩"
        ]
    },
    # "ddtn": {
    #     "merge": {
    #         '片条块': ['西瓜', '柚子皮', '玉米棒', '香蕉', '片条块'], None: ['鸡蛋', '蛋壳'], '瓶类': ['饮料瓶', '瓶类'],
    #         '垃圾桶-其他果蔬': ['垃圾桶-柚子皮', '垃圾桶-其他果蔬'],
    #         '垃圾桶-大块果蔬皮': ['垃圾桶-橘橙', '垃圾桶-柚子皮', '垃圾桶-大块果蔬皮']
    #     },
    # },
    "ddtn_eight": {
        "merge": {
            '片条块': ['西瓜', '柚子皮', '玉米棒', '香蕉', '片条块'], '瓶类': ['饮料瓶', '瓶类']
        },
    },
    "ddtn_thirty-six": {
        "merge": {
            '垃圾桶-散碎': [
                '垃圾桶-散碎', '垃圾桶-红薯皮', '垃圾桶-香菜', '垃圾桶-茼蒿', '垃圾桶-空心菜', '垃圾桶-土豆', '垃圾桶-芹菜', '垃圾桶-葱青', '垃圾桶-豆角',
                '垃圾桶-梨', '垃圾桶-毛豆', '垃圾桶-韭菜', '垃圾桶-条状青菜'
            ],
            '垃圾桶-纸巾纸张': ['卫生巾', '尿不湿', '垃圾桶-纸巾纸张', '垃圾桶-尿不湿', '垃圾桶-卫生巾'], '垃圾桶-西瓜': ['西瓜', '垃圾桶-西瓜'],
            '垃圾桶-香蕉': ['香蕉', '垃圾桶-香蕉'], '垃圾桶-橘橙': ['橘橙', '垃圾桶-橘橙'], '垃圾桶-花生': ['花生', '垃圾桶-花生'],
            '垃圾桶-玉米': ['玉米棒', '玉米皮', '垃圾桶-玉米'], '垃圾桶-白菜': ['白菜', '垃圾桶-白菜'], '垃圾桶-瓜子': ['瓜子', '垃圾桶-瓜子'],
            '垃圾桶-中药渣': ['药渣', '垃圾桶-中药渣'], '垃圾桶-柚子皮': ['柚子皮', '垃圾桶-柚子皮'], '垃圾桶-剩菜剩饭': ['剩菜剩饭', '垃圾桶-剩菜剩饭'],
            '垃圾桶-火龙果': ['火龙果', '垃圾桶-火龙果'], '垃圾桶-苹果': ['苹果', '垃圾桶-苹果'],
            '垃圾桶-其他果蔬': ['垃圾桶-芒果', '垃圾桶-南瓜', '垃圾桶-洋葱', '垃圾桶-红薯', '垃圾桶-其他果蔬', '垃圾桶-茄子'],
            '垃圾桶-菜叶': ['垃圾桶-小青菜', '垃圾桶-生菜', '垃圾桶-菠菜', '垃圾桶-菜叶'], '垃圾桶-口罩': ['口罩', '垃圾桶-口罩'],
            '垃圾桶-香烟头': ['香烟头', '垃圾桶-香烟头'], '垃圾桶-玻璃瓶': ['酒瓶', '垃圾桶-玻璃瓶'], '垃圾桶-易拉罐': ['易拉罐', '垃圾桶-易拉罐'],
            '垃圾桶-其他盒状': ['垃圾桶-药盒', '垃圾桶-其他盒状'], '垃圾桶-其他包装袋': ['垃圾桶-零食包装袋', '垃圾桶-其他包装袋'],
            '垃圾桶-石榴': ['石榴', '垃圾桶-石榴'], '垃圾桶-药片': ['药片', '垃圾桶-药片'], '垃圾桶-核桃': ['核桃', '垃圾桶-核桃', '核桃皮'],
            '垃圾桶-番茄': ['番茄圣女果'], '垃圾桶-蒜': ['大蒜', '垃圾桶-蒜皮', '蒜皮', '垃圾桶-蒜']
        }
    },
    "waste": {
        "merge": {
            "waste": [
                '垃圾桶-中药渣', '垃圾桶-其他包装袋', '垃圾桶-其他盒状', '垃圾桶-剩菜剩饭', '垃圾桶-南瓜', '垃圾桶-卫生巾', '垃圾桶-口罩', '垃圾桶-哈密瓜', '垃圾桶-土豆',
                '垃圾桶-塑料瓶', '垃圾桶-塑料袋', '垃圾桶-奶盒', '垃圾桶-奶袋', '垃圾桶-小青菜', '垃圾桶-尿不湿', '垃圾桶-散碎', '垃圾桶-易拉罐', '垃圾桶-条状青菜',
                '垃圾桶-柚子皮', '垃圾桶-核桃', '垃圾桶-梨', '垃圾桶-橘橙', '垃圾桶-毛豆', '垃圾桶-水果泡沫网', '垃圾桶-洋葱', '垃圾桶-火龙果', '垃圾桶-玉米', '垃圾桶-玻璃瓶',
                '垃圾桶-瓜子', '垃圾桶-生菜', '垃圾桶-番茄', '垃圾桶-白菜', '垃圾桶-石榴', '垃圾桶-空心菜', '垃圾桶-红薯', '垃圾桶-红薯皮', '垃圾桶-纸巾纸张', '垃圾桶-芒果',
                '垃圾桶-花生', '垃圾桶-芹菜', '垃圾桶-苹果', '垃圾桶-茄子', '垃圾桶-茼蒿', '垃圾桶-药片', '垃圾桶-药盒', '垃圾桶-菜叶', '垃圾桶-菠菜', '垃圾桶-葡萄',
                '垃圾桶-葱青', '垃圾桶-蒜', '垃圾桶-蒜皮', '垃圾桶-西瓜', '垃圾桶-豆角', '垃圾桶-零食包装袋', '垃圾桶-韭菜', '垃圾桶-香烟头', '垃圾桶-香菜', '垃圾桶-香蕉',
                '垃圾桶-鸡蛋', '垃圾桶-多品类', '垃圾桶-果蔬块', '垃圾桶-空', '垃圾桶-侧', '垃圾桶-闭合'
            ],
            "other": [
                'kindle', 'x光片', '一次性塑料浴帽', '一次性塑料盘', '一次性塑料调羹', '一次性拖鞋', '一次性棉签', '一次性餐盒', '丝瓜', '乒乓球', '乒乓球拍', '书架',
                '书纸', '书纸杂志', '传单', '体重秤', '便利贴', '保温杯', '保险箱', '保鲜盒', '修正带', '充气沙发', '充电头', '充电宝', '充电线', '光盘', '八宝粥',
                '八宝粥罐', '农药瓶', '冬瓜', '冰柜', '冰激凌', '冰糖葫芦', '冻豆腐', '净水器', '凉席', '剪刀', '包菜', '化妆品瓶', '午餐肉', '单开门冰箱',
                '单排旱冰鞋', '南瓜', '占位符', '卡片', '卫生巾', '卸甲水', '双排旱冰鞋', '发箍', '变形玩具', '口服液', '口罩', '台灯', '吸顶灯', '吹风机', '呼啦圈',
                '咖啡', '咖啡机', '哈密瓜甜瓜', '哑铃', '围裙', '圆规', '土豆', '土豆丝', '地球仪', '地铁卡车票', '塑料扣', '塑料杯纸杯', '塑料瓶', '塑料瓶瓶盖',
                '塑料盒', '塑料碗盆', '塑料袋', '塑料袋装牛奶', '墨镜', '大葱', '大蒜', '大骨头', '太阳能热水器', '头盔', '奶盒', '奶粉罐食品罐', '姜', '姜片',
                '婴儿床', '安全套包装袋', '宠物饲料包装袋', '室内垃圾包', '宽粉', '小提琴', '小米', '小青菜', '尖椒', '尺子', '尿不湿', '山楂', '山竹', '山药',
                '巧克力', '巴旦木', '布条', '常见肉类', '帽子', '干燥剂', '干面条', '平菇', '床', '床单被套', '床头柜', '废弃电线', '开关', '开心果', '开瓶器',
                '开罐器', '彩椒', '微波炉', '快递袋信封', '手套', '手持吸尘器', '手持电风扇', '手机', '手机壳', '手电筒', '手表', '手链', '打印机', '打印机墨盒',
                '打气筒', '扫地机器人', '扫把', '报纸', '披萨', '抽油烟机', '抽纸塑料袋', '拖把', '排球', '排骨', '插头电线', '插线板', '搓澡巾', '搪瓷碗', '擀面杖',
                '收音机', '放大镜', '方便面调味包', '施工安全帽', '无花果', '日历1', '旧衣服', '易拉罐', '显示屏', '晾衣架', '暖宝宝帖', '望远镜', '木制玩具',
                '木头铅笔', '木桶', '木梳子', '木瓜', '木组合沙发', '木质衣架', '木铲', '木雕', '机箱', '杀虫剂', '杂志', '李子', '杨梅', '板栗', '枕头',
                '染发剂罐', '柚子皮', '柠檬', '核桃', '桂圆', '桂花糕', '桃子', '桌子', '档案袋', '棉签盒', '棋子_跳棋', '棒棒糖', '椅凳', '榨汁机', '榴莲壳',
                '槟榔', '樱桃车厘子', '橘橙', '止痛膏', '毛巾抹布', '毛毯', '水壶', '水银温度计', '水饺', '汉堡', '汽车牌照', '油条', '油漆笔', '油麦菜', '泡沫箱盒',
                '泡腾片', '泡菜', '泡面袋', '波波肠', '波轮洗衣机', '洋葱', '洗发水瓶', '洗衣液', '测试卷子', '浴缸', '海绵粉扑', '消毒剂', '混沌', '游戏手柄',
                '湿纸巾', '滑冰鞋', '滚筒洗衣机', '滴眼液瓶', '漆胶桶', '火腿肠', '火锅', '火龙果', '灭火器', '炒锅', '炒饭', '炸鸡', '烟灰缸', '烟盒', '烤肉',
                '烤肠', '烧烤锡纸', '热水瓶', '热水袋', '照片', '照相机', '照相机电池', '煮好的方便面', '熟里脊肉', '燃气灶', '燃气热水器', '燃气瓶', '燕麦片', '牙刷',
                '牙签', '牛肉干', '牛肉牛排', '狗粮', '猫砂', '玉米棒', '玉米粒', '玩具玩偶', '玻璃', '玻璃壶', '玻璃杯罐', '玻璃球', '珍珠', '瑜伽球', '瓜子',
                '甘蔗', '生煎包子', '生菜', '生菜包菜', '电动剃须刀', '电动卷发棒', '电子烟烟弹', '电子秤', '电子血压仪', '电子闹钟', '电池', '电池板', '电热毯',
                '电热水器', '电熨斗', '电磁炉', '电视遥控器', '电路板', '电风扇', '电饭煲', '番茄圣女果', '番茄酱包装袋', '登机牌', '白板笔', '白菜', '白萝卜', '百洁布',
                '皮夹卡包', '皮布单人沙发', '皮布组合沙发', '皮带', '皮蛋', '皮鞋', '盘子', '相框', '相片纸', '眼线笔', '眼镜', '眼镜布', '石榴', '砂锅', '碎瓷片',
                '碗碟', '碘酒瓶', '碧根果', '磁铁', '空心菜', '空气净化器', '空气加湿器', '空调外机', '空调挂机', '空调柜机', '空调滤芯', '空调遥控器', '窗帘', '竹笋',
                '竹签', '笔', '笔芯', '笔记本电脑', '笼子', '筷子', '篮球', '粉条', '粘鼠胶', '糕点饼类', '紫菜', '紫薯地瓜', '红外电子体温计', '红枣', '红肠',
                '红花油', '红酒塞', '纸', '纸巾', '纸巾包装袋', '纸巾盒', '纸牌', '纸箱', '纸袋', '纽扣电池', '绳子', '绿豆', '绿豆芽', '网卡', '网球', '羽毛球',
                '羽毛球拍', '老式电视机', '老式电话', '耳套', '耳机', '耳钉耳环', '肥牛片', '肯德基纸袋', '背包手提包', '胡萝卜', '胶带', '胶棒', '胶水', '腊肠',
                '腋下电子温度计', '腐竹', '自行车', '节能灯泡', '芋头', '芒果', '芦荟', '芭比娃娃', '花椒', '花洒', '花洒软管', '花生', '花盆', '花菜', '芹菜',
                '苍蝇拍', '苦瓜', '苦菊', '苹果', '茶叶', '茶叶罐', '茶叶蛋', '茶壶', '茼蒿', '草帽', '草莓', '荔枝', '荧光灯', '荧光笔', '药渣', '药片',
                '药瓶', '药盒药袋', '药袋', '荷包蛋', '莲藕', '莴笋', '菜刀', '菜板', '菠菜', '菠萝', '菠萝包', '菠萝蜜', '落叶', '葡萄', '蒜薹', '蒸锅',
                '蓄电池', '薄荷叶', '薯条', '薯片', '藕片', '虾', '蚊香片', '蚕豆', '蛋壳', '蛋挞', '蛋糕', '蛋糕盒', '蛋黄酥', '蜡笔', '螺丝刀', '螺钉',
                '行李箱', '行车记录仪', '衣服吊牌', '衣柜', '袜子', '裙子', '裤子', '西兰花', '西柚', '西瓜', '西瓜子', '计算器', '订书机', '话梅', '话筒',
                '调料瓶', '豆浆机', '豆腐', '豆腐皮', '豆角', '贝壳', '贝果', '跑步机', '路由器', '轮椅', '轮胎', '软膏', '输液瓶', '辣条包装袋', '辣椒', '酒瓶',
                '酒精灯', '量杯', '金属碗盆', '金属门吸', '金桔', '金针菇', '针头', '钉子', '钢丝球', '钥匙钥匙扣', '银耳', '锅', '锅盖', '锤子', '键盘', '镊子',
                '长尾夹', '长椅', '闹钟', '陈皮', '雨伞', '青茄子', '青豆豌豆', '面包', '鞋', '韭菜', '音响', '食用油桶', '餐具_刀叉勺', '餐垫', '饮料瓶',
                '饮水机', '饼干', '馒头', '首饰盒', '香烟头', '香菇', '香菜', '香蕉', '香蕉干', '马克杯', '验孕棒', '骨肉相连', '高压锅', '鱼缸', '鱼肉', '鱼骨',
                '鲜枣', '鸡毛掸', '鸡翅鸡肉', '鸡胸肉', '鸡蛋', '鸡蛋包装盒', '鸡骨', '鹌鹑蛋', '麻将台', '黄瓜', '黄豆', '黑木耳', '黑豆', '鼠标', 'voc',
                'unknown', '清晰多品类', '办公桌', '起居室', '厨房', '卫生间', '桌面', '墙板地面', '户外'
            ],
        },
    },
    "ddtn_30": {
        "merge": {
            '垃圾桶-厨余散碎': [
                '垃圾桶-剩菜剩饭', '垃圾桶-散碎'
            ],
            '垃圾桶-条状青菜': ['垃圾桶-芹菜', '垃圾桶-葱青', '垃圾桶-韭菜', '垃圾桶-香菜', '垃圾桶-豆角', '垃圾桶-茼蒿', '垃圾桶-条状青菜', '垃圾桶-空心菜'],
            '垃圾桶-菜叶': ['垃圾桶-小青菜', '垃圾桶-菠菜', '垃圾桶-生菜', '垃圾桶-菜叶'],
            '垃圾桶-果蔬皮': ['垃圾桶-红薯皮', '垃圾桶-土豆皮', '垃圾桶-梨皮', '垃圾桶-毛豆', '垃圾桶-苹果皮', '垃圾桶-果蔬皮'],
            '垃圾桶-果蔬块': ['垃圾桶-洋葱', '垃圾桶-南瓜', '垃圾桶-红薯', '垃圾桶-茄子', '垃圾桶-芒果', '垃圾桶-哈密瓜', '垃圾桶-火龙果', '垃圾桶-石榴', '垃圾桶-果蔬块', '垃圾桶-苹果'],
            '垃圾桶-奶盒奶袋': ['垃圾桶-奶盒', '垃圾桶-奶袋'],
            '垃圾桶-包装袋': ['垃圾桶-其他包装袋', '垃圾桶-零食包装袋'],
            '垃圾桶-盒状': ['垃圾桶-其他盒状', '垃圾桶-药盒'],
            '垃圾桶-瓶类': ['垃圾桶-塑料瓶', '垃圾桶-玻璃瓶', '垃圾桶-易拉罐'],
            '垃圾桶-纸巾纸张': ['垃圾桶-尿不湿', '垃圾桶-卫生巾', '尿不湿', '卫生巾'],
            '垃圾桶-蒜': ['垃圾桶-蒜皮', '垃圾桶-蒜'],
            '垃圾桶-剩菜剩饭': ['剩菜剩饭', '垃圾桶-剩菜剩饭'],
            '垃圾桶-瓜子': ['瓜子', '垃圾桶-瓜子'],
            '垃圾桶-药片': ['垃圾桶-药片', '药片'],
            '垃圾桶-口罩': ['垃圾桶-口罩', '口罩'],
            '垃圾桶-香烟头': ['垃圾桶-香烟头', '香烟头'],
            '垃圾桶-柚子皮': ['垃圾桶-柚子皮', '柚子皮'],
            '垃圾桶-核桃': ['垃圾桶-核桃', '核桃', '核桃皮'],
            '垃圾桶-番茄': ['垃圾桶-番茄', '番茄圣女果'],
            '垃圾桶-易拉罐': ['易拉罐', '垃圾桶-易拉罐'],
            '垃圾桶-花生': ['垃圾桶-花生', '花生'],
            '垃圾桶-西瓜': ['垃圾桶-西瓜', '西瓜'],
            '垃圾桶-香蕉': ['垃圾桶-香蕉', '香蕉'],
            '垃圾桶-白菜': ['垃圾桶-白菜', '白菜'],
            '垃圾桶-橘橙': ['垃圾桶-橘橙', '橘橙'],
            '垃圾桶-玉米': ['垃圾桶-玉米', '玉米棒', '玉米皮'],
            '垃圾桶-中药渣': ['垃圾桶-中药渣', '药渣']
        }
    },
}


def gen_merge_data(data, project_name, extra_map=None):
    """

    :param data: {class_name: [图片的路径, ]}
    :param project_name: 项目名：dddp
    :param extra_map: {key: num}  做合并的时候每一类只需要多少张图片
    :return:
    """
    extra_map = extra_map or {}
    merge = merge_data.get(project_name, {})
    # print(merge)

    for old_name, new_name in merge.get("rename", {}).items():
        cur_data = data.pop(old_name, [])
        data.setdefault(new_name, []).extend(cur_data)

    for main_class_name, merge_class_name_list in merge.get("merge", {}).items():
        for merge_class_name in merge_class_name_list:

            if main_class_name in extra_map:
                cur_data = data.pop(merge_class_name, [])[:extra_map[main_class_name]]
            else:
                cur_data = data.pop(merge_class_name, [])
            data.setdefault(main_class_name, []).extend(cur_data)

    for rm_class_name in merge.get("remove", []):
        data.pop(rm_class_name, [])

    return data
