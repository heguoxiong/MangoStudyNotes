#### 目标跟踪

目标跟踪研究综述

<img src="https://pic2.zhimg.com/v2-632a3a08c0f30f0abcdb8b06afbe346d_b.jpg" alt="img"  />

#### YOLO

1. 环境配置

   ```
   # 进入pytorch终端，输入命令
   pip install -r requirements.txt
   # 针对pycocotools问题，输入命令
   pip install pycocotools
   ```

   错误1

   ![image-20220414154057932](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414154057932.png)

   - 输入命令：pip install -r requirements.txt

   错误2

   ![image-20220414154417941](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414154417941.png)

   - 需要下载C++编译工具：https://visualstudio.microsoft.com/visual-cpp-build-tools/

   ​	![image-20220414155045709](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414155045709.png)

   ![image-20220414155136638](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414155136638.png)

   - 输入命令：pip install pycocotools

2. 运行"detect.py"报错

   ![image-20220414192111289](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414192111289.png)

   解决方法：在"model/common.py"中加入下面的代码

   ```
   import warnings
    
    
    
   class SPPF(nn.Module):
       # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
       def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
           super().__init__()
           c_ = c1 // 2  # hidden channels
           self.cv1 = Conv(c1, c_, 1, 1)
           self.cv2 = Conv(c_ * 4, c2, 1, 1)
           self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
       def forward(self, x):
           x = self.cv1(x)
           with warnings.catch_warnings():
               warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
               y1 = self.m(x)
               y2 = self.m(y1)
               return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
   ```

   

3. "detect.py"文件参数介绍

   ![image-20220414195144915](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414195144915.png)

   ![image-20220414211636623](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414211636623.png)

   模型训练使用的图片尺寸

   ![image-20220414195800785](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414195800785.png)

   训练"img-size"与输入输出img尺寸

   ![image-20220414200036976](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414200036976.png)

4. 模型训练"train.py"

   执行"train.py"，模型输出如下

   ![image-20220415171811486](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415171811486.png)

   模型训练使用的数据集

   ![image-20220415172342808](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415172342808.png)

   模型参数配置数据

   ![image-20220415172533551](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415172533551.png)

   模型超参数

   ![image-20220415172747491](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415172747491.png)

   main函数参数介绍

   ![image-20220415195729686](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415195729686.png)

5. 自制数据集训练

   - 数据集获取方式

     人工标注；仿真数据集（GAN）

   - 在线标注工具：https://www.makesense.ai/

     ![image-20220415202918207](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415202918207.png)

   - 创建训练文件的目录结构

     ![image-20220415203915671](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415203915671.png)

   - 添加数据集的配置文件

     ![image-20220415204050447](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415204050447.png)

#### 在anaconda创建pytorch环境

1. 打开anaconda prompt，输入命令

   ```
   # 环境名称：pytorchhgx 依赖的python版本：3.6 可以在"Anaconda3\envs\"目录下查看环境文件
   conda create -n pytorchhgx python=3.6
   # 激活环境
   conda activate pytorchhgx
   # 取消激活
   conda deactivate
   # 查看当前环境有哪些包
   pip list/conda list
   # 查看驱动版本。可以查看“驱动”、“cuda"版本信息
   nvidia-smi
   # 检查pytoch是否安装成功
   python
   import torch
   # 检查pytorch是否可以使用GPU
   torch.cuda.is_available()
   # 查看已有的环境
   conda info --envs
   # 删除创建的虚拟环境
   conda remove -n 环境名 --all
   ```

2. 查看电脑的显卡是否支持cuda

   方法1：进入网址https://www.geforce.cn/hardware/technology/cuda/supported-gpus

   方法2：打开NVIDIA控制面板->系统信息->组件

   依据显卡是否支持cuda，在pytorch官网选择对应的pytorch安装命令。

3. 安装python编辑器

   pycharm

   jupyter（交互式）

   ```
   # 在pytorch环境中安装jupyter
   进入pytorch环境，安装依赖
   conda install nb_conda
   ```


#### Git分布式版本控制

1.工作机制及托管中心介绍

![image-20220413211940911](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413211940911.png)

代码托管中心是基于网络服务器的远程代码仓库，简称远程库。（push）

![image-20220413212147515](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413212147515.png)

2.Git常用命令

![image-20220413213528681](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413213528681.png)

```
# 设置用户签名
只需设置一次，但必须设置，否则无法提交代码。执行完下列命令，可以在用户加目录"C:\Users\86153"查看到".gitconfig"的文件。签名的作用是区分不同操作者身份，用户的签名信息在每一个版本的提交信息中能够看到；这里设置用户签名和登录Github（或其他代码托管中心）的账号没有任何关系。
git config --global user.name mango
git config --global user.email virtual@.com
# 初始化本地库
git init
# 查看本地库状态
git status
# 文件添加到暂存区
git add 文件名
# 将文件从暂存区删除(本地目录的工作区中的文件还在)
git rm --cached hello.txt
# 将暂存区的文件提交到本地库
git commit -m "日志信息" 文件名
# 查看精简历史信息(精简版本号、分支、日志)
git reflog
# 查看详细历史信息（详细版本号、提交者信息、提交日期）
git log
# 版本穿梭（通过HEAD指向的版本号，实现向前穿梭、向后穿梭）
git reset --hard 版本号
```

查看本地库状态

![image-20220413215152727](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413215152727.png)

在工作区新建一个hello.txt文件后（红色表示文件未被追踪！）

![image-20220413215906897](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413215906897.png)

将hello.txt提交到暂存区。将文件从**暂存区**（工作区中文件还在）中删除"git rm --cached hello.txt"

![image-20220413220236521](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413220236521.png)

将文件从暂存区提交到本地库

![image-20220413221018951](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413221018951.png)

查看提交信息

![image-20220413221151821](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413221151821.png)

查看提交者信息

![image-20220413221449742](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413221449742.png)

**修改已提交到本地库的文件**（修改后红色表示此时文件未被追踪）

![image-20220413221737946](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413221737946.png)

将修改后的文件**再次**提交到暂存区

![image-20220413222031225](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413222031225.png)

从暂存区中，将修改后的文件再次提交到本地库（指针指向第二个版本！）

![image-20220413222248964](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413222248964.png)

从第二个版本回到第一个版本（也可以从第一个版本再次返回第二个版本！）

![image-20220413222801116](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413222801116.png)

版本穿梭示意图

![image-20220413223221583](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220413223221583.png)

3.Git_分支

![image-20220414094111825](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414094111825.png)

![image-20220414094449623](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414094449623.png)

![image-20220414094655806](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414094655806.png)

```
# 查看分支
git branch -v
# 创建分支
git branch 分支名
# 切换分支(工作区文件对应当前指向的分支)
git checkout 分支名
# 合并分支(把指定的分支合并到当前分支上)
git merge 分支名
```

查看分支

![image-20220414095048345](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414095048345.png)

创建分支

![image-20220414095218535](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414095218535.png)

切换分支

![image-20220414095435420](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414095435420.png)

合并分支

![image-20220414100031997](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414100031997.png)

冲突合并（手动编辑冲突文件进行修改）

![image-20220414100457965](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414100457965.png)

![image-20220414100642787](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414100642787.png)

合并分支只会修改合并的分支。修改完冲突代码后，需要重新提交文件；而且commit不能带文件名

![image-20220414100917635](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414100917635.png)

4.Git_团队协作

团队内协作

![image-20220414101421108](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414101421108.png)

跨团队协作

![image-20220414101710297](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414101710297.png)

5.Git_Github

![image-20220414112243907](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414112243907.png)

```
# 查看当前远程库别名
git remote -v
# 给远程库起别名
git remote add 别名 远程地址 # 远程地址是http链接
# git命令行中的复制快捷键
Shift+Insert
# 推送本地库分支内容到远程仓库（需要验证账号）
git push 别名 分支
# 将远程仓库对于分支最新内容拉下来后与当前本地分支直接合并。拉取动作会自动提交到本地库
git pull 远程库地址别名 远程分支名
# 针对上传和拉取问题，有时需要关闭ssl验证
git config --global http.sslVerify "false"
# 将远程仓库的内容克隆到本地(不需要验证账号)：1拉取代码；2初始化本地库；3远程库取别名为"origin"
git clone 远程地址
```

给远程库创建别名

![image-20220414113301667](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414113301667.png)

**推动本地库分支内容到远程库**

- 打开“凭据管理器”，查看是否有证书凭据

  ![](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414113913859.png)

- 授予许可：授权后就会增加github的普通凭据

  ![image-20220414152143504](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414152143504.png)

- 推送成功

  ![image-20220414114234388](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414114234388.png)

**拉取远程库到本地库**

- Github网页对文件进行编辑修改并"commit"后，如何拉取到本地库

- 接触SSL验证后，拉取成功（拉取动作会自动提交到本地库）

  ![image-20220414115914299](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414115914299.png)

**将远程库克隆到本地**

![image-20220414155844606](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414155844606.png)

​	克隆会自动给远程库取别名为"origin"

![image-20220414160047527](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414160047527.png)

Git团队内协作：A账号push的代码。B账号克隆下来进行编辑修改后，再push到A的github项目中。

- 需要在A的github项目中，添加B为团队成员

  ![image-20220414161211379](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414161211379.png)

  ​	A账号添加邀请B，生成邀请链接发送给B账号

  ![image-20220414161409472](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414161409472.png)

  ​	B账号接收A的团队邀请，就可以push代码到远程库！

  ![image-20220414161641865](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414161641865.png)

Git跨团队协作：fork()

![image-20220414201653843](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414201653843.png)

SSH免密登录：设置ssh后，可以免密拉取

```
# 进入到用户家目录"C:\Users\86153"打开Git，输入命令
ssh-keygen -t rsa -C 2965531503@qq.com #邮箱时Github账号的邮箱
```

​	生成".ssh"文件夹，包含公钥和私钥

![image-20220414163214386](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414163214386.png)

​	将公钥内容添加到Github账号中

![image-20220414163516229](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414163516229.png)

![image-20220414163711037](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220414163711037.png)

问题：

![image-20220416210159302](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416210159302.png)

解决方法：

​	进入.git目录下的config文件

![image-20220416210314147](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416210314147.png)

​	将url地址改为ssh地址

![image-20220416210411310](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416210411310.png)

#### vim命令

```
# 复制（退出插入模式）
yy
# 粘贴（退出插入模式）
p
```

#### Linux命令

```
# 查看文件中的内容
cat hello.txt
# 查看文件中最后一行的内容
tail -n 1 hello.txt
# 查看目录下各文件信息
ll
```



#### VSCode

1. Windows搭建vscode

   - 开发环境搭建

     - 安装mingw-w64编译器（GCC for Windows）、CMake工具（选装）

       The mingw-w64 project is a complete runtime environment for **gcc** to support binaries native to Windows 64-bit and 32-bit operating systems.

       ![image-20220415232224840](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220415232224840.png)

       gcc：C语言编译器

       g++：C++语言编译器

     - vscode插件安装

       - C/C++
       - cmake（选装）
       - cmake tools（选装）

   - 代码实践演练

     - 基于g++的命令

       ```
       # 使用g++编译cpp文件，生成a.exe可执行文件
       g++ main.cpp
       # 生成可调试为文件(-g)；同时指定可执行文件名称为"myname.exe"(-o)
       g++ -g .\main.cpp -o myname
       # 多个cpp文件参与编译
       g++ -g .\main.cpp .\swap.cpp -o myname
       ```

       调试

       ![image-20220416212500322](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416212500322.png)

       ![image-20220416212613016](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416212613016.png)

       创建launch.json：指定gdb工具和待调试的exe文件。针对调试(debug)和启动(run)的配置项。

       ![image-20220416213826912](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416213826912.png)

       ![image-20220416220822853](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416220822853.png)

       

       .json文件中相关变量的介绍

       ```
       ${workspaceFolder}
       ```

       ![image-20220416220515718](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416220515718.png)

     - 基于cmake

       方法一："Ctrl+Shift+P"：需要为cmake工具配置"编译器"环境！（gcc/vs）

       ![image-20220416221604809](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416221604809.png)

       ![image-20220416221640438](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220416221640438.png)

       进入到build文件夹，执行cmake

       ```
       # 进入到build文件夹，执行cmake
       cd .\build\
       cmake ..
       # 执行windows下的编译程序（相当于make）
       mingw32-make
       # 运行编译好的程序
       .\my_cmake_swap.exe
       ```

       配置launch.json中可执行文件的路径信息后，即可开始调试此程序

       ![image-20220419174426034](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419174426034.png)

       方法二：自己建立build文件夹，命令行执行cmake

       ```
       # 创建build文件夹，进入build文件夹
       mkdir build
       cd build
       # 如果电脑上已经安装了VS，可能会调用微软的MSVC编译器。可以使用（cmake -G "MinGW Makefiles" ..)代替（cmake ..）
       cmake -G "MinGW Makefiles" ..
       # 开始编译
       mingw32-make
       ```

       ![image-20220419175532895](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419175532895.png)

     - 配置tasks.json：通过在launch.json中关联tasks.json，在每次更新代码后，会根据tasks.json中的配置重新执行编译

       ![image-20220419202228759](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419202228759.png)

       将launch.json和tasks.json关联起来

       ![image-20220419203242460](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419203242460.png)

       ![image-20220419204551887](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419204551887.png)

     - 针对luanch.json需要注意的点

       ![image-20220419205521850](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419205521850.png)

     - 利用tasks.json配置cmake

       ![image-20220419211640913](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419211640913.png)

       ![image-20220419211731061](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419211731061.png)

     - luanch.json样本

       ```
       {
           // 使用 IntelliSense 了解相关属性。 
           // 悬停以查看现有属性的描述。
           // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
           "version": "0.2.0",
           "configurations": [
               {
                   "name": "(gdb) 启动",
                   "type": "cppdbg",
                   "request": "launch",
                   "program": "${workspaceFolder}/out.exe",
                   "args": [],  //此处的参数会传送给main函数中的argv中
                   "stopAtEntry": false,
                   "cwd": "${workspaceFolder}",  //进入该目录下
                   "environment": [],
                   "externalConsole": false,
                   "MIMode": "gdb",
                   "miDebuggerPath": "G:\\softInstall\\mingw64\\bin/gdb.exe",
                   "setupCommands": [
                       {
                           "description": "为 gdb 启用整齐打印",
                           "text": "-enable-pretty-printing",
                           "ignoreFailures": true
                       },
                       {
                           "description":  "将反汇编风格设置为 Intel",
                           "text": "-gdb-set disassembly-flavor intel",
                           "ignoreFailures": true
                       }
                   ],
                   "preLaunchTask": "Compile_url"
               }
           ]
       }
       ```

     - tasks.json样本

       样本一

       ```
       {
           // See https://go.microsoft.com/fwlink/?LinkId=733558
           // for the documentation about the tasks.json format
           // tasks.json这个文件是定义调试开始前要执行的任务，即（或者绝大多数是）编译程序， 定义了用于编译程序的编译器，所输出的文件格式，使用的语言标准等
           "version": "2.0.0",
           "tasks": [
               {
                   "label": "Compile_url", // 任务名称，与launch.json的preLaunchTask相对应
                   "command": "G:\\softInstall\\mingw64\\bin\\g++.exe", // 要使用的编译器, C就写gcc
                   "args": [
                       //"${file}",
                       "main.cpp",
                       "swap.cpp",
       
                       "-o", // 指定输出文件名，不加该参数则默认输出a.exe，Linux下默认a.out
                       //"${fileDirname}/${fileBasenameNoExtension}.out",
                       "${workspaceFolder}/out.exe",
       
       
                       "-g", // 生成和调试有关的信息
                       //"-Wall", // 开启额外警告
                       //"-static-libgcc", // 静态链接
                       //"-std=c11" // C语言最新标准为c11，或根据自己的需要进行修改比如C++17
                   ], // 编译命令参数
                   "type": "shell", // 可以为shell或process，前者相当于先打开shell再输入命令，后者是直接运行命令
                   "group": {
                       "kind": "build",
                       "isDefault": true // 设为false可做到一个tasks.json配置多个编译指令，需要自己修改本文件，我这里不多提
                   },
                   "presentation": {
                       "echo": true,
                       "reveal": "always", // 在“终端”中显示编译信息的策略，可以为always，silent，never。具体参见VSC的文档
                       "focus": true, // 设为true后可以使执行task时焦点聚集在终端
                       "panel": "shared" // 不同的文件的编译信息共享一个终端面板
                   },
                   //"problemMatcher": "$gcc"
               }
           ]
       }
       ```

       样本二：使用tasks.json配置cmake

       ```
       {
           // See https://go.microsoft.com/fwlink/?LinkId=733558
           // for the documentation about the tasks.json format
           // tasks.json这个文件是定义调试开始前要执行的任务，即（或者绝大多数是）编译程序， 定义了用于编译程序的编译器，所输出的文件格式，使用的语言标准等
           "version": "2.0.0",
           "options": {
               "cwd": "${workspaceFolder}/build" //进入到build文件夹
           },
           "tasks": [
               {
                   "label": "cmake", // 任务名称，与launch.json的preLaunchTask相对应
                   "command": "cmake", // 要使用的编译器, C就写gcc
                   "args": [
                       "..",
                   ], // 编译命令参数
                   "type": "shell" // 可以为shell或process，前者相当于先打开shell再输入命令，后者是直接运行命令
               },
               {
                   "label": "make", // 任务名称
                   "command": "mingw32-make", // linux下是make
                   "args": [], // 编译命令参数
                   "group": {
                       "kind": "build",
                       "isDefault": true // 设为false可做到一个tasks.json配置多个编译指令，需要自己修改本文件，我这里不多提
                   }
               },
               {
                   "label": "Compile_url", // 任务名称
                   "dependsOn":[
                       "cmake",
                       "make"
                   ]
               }
           ]
       }
       ```

       

     - 资料：vscode中的预定义变量

       ![image-20220419203932074](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419203932074.png)

       示例

       ![image-20220419204116262](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419204116262.png)

2. vscode版本控制

   官方文档：https://code.visualstudio.com/docs/editor/versioncontrol

3. vscode设置

   设置代码编辑按行自动格式化：文件->首选项->设置    进入后打开Editor:Format on Type

   ![image-20220419181137306](C:\Users\86153\AppData\Roaming\Typora\typora-user-images\image-20220419181137306.png)

#### cmake(黑马程序员make)

Linux默认安装库的位置；头文件的位置



