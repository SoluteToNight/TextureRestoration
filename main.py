# 这是一个示例 Python 脚本。
import workflow
import os
import ntpath as path

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    scripts_path = os.getcwd()
    obj_folder = "obj"
    folders = os.listdir(obj_folder)
    fd_path = []
    for fd in folders:
        fd_path.append(os.path.join(obj_folder, fd))
    img_dir = os.listdir(fd_path[0])
    img_dir = list(filter(lambda x: x.rfind(".png") != -1, img_dir))
    # img_dir  = [path for path in img_dir if path.rfind(".png") != -1]
    for i in range(len(img_dir)):
        img_dir[i] = os.path.join(scripts_path,fd_path[0], img_dir[i])
    for img in img_dir:
        flow = workflow.analyse(img)
        result = flow.process()
        print(result)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
