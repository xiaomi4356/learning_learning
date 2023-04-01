import matplotlib.pyplot as plt
from matplotlib import font_manager

for font in font_manager.fontManager.ttflist:
    # 查看字体名以及对应的字体文件名
    print(font.name, '-', font.fname)