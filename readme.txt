1.初始化操作
(1) 进入caffe根目录(本文中caffe的根目录都为caffe-root)，创建临时文件夹，用于存放所需要的临时文件
	mkdir examples/_temp

(2) 将视频转为一帧帧图片并将图片复制到 examples/images 文件夹下
	python VideoToImg.py 
		
(3) 根据examples/images文件夹中的图片，创建包含图像列表的txt文件，并添加标签（0）
	find `pwd`/examples/images -type f -exec echo {} \; > examples/_temp/temp.txt
	或者 python getList.py

(4) 执行下列脚本，下载imagenet12图像均值文件，在后面的网络结构定义prototxt文件中，需要用到该文件		  （data/ilsvrc212/imagenet_mean.binaryproto）.下载模型以及定义prototxt。
	sh ./data/ilsvrc12/get_ilsvrc_aux.sh
(5) 将网络定义prototxt文件复制到_temp文件夹下, 并修改 batchsize
	cp examples/feature_extraction/imagenet_val.prototxt examples/_temp
		
(6) 批量提取图片特征
	sh extract_fea.sh		
	( 输入:存有若干图片的文件夹路径 ; 输出:每张图片fc7输出的特征向量vec[i] )
	起源于前几天想用for循环去forward处理每张图片,但得到的结果不正确
	原因在于进行forward过程时一个batch对应的blobs数据在被存储前就被下一个batch
	函数:
	1 ) feat_helper_pb2.py : 保存每层输出的数据为lmdb格式
	2 ) lmdb2mat.py : 将lmdb数据转化成.mat数据
	3 ) extract_fea.sh : 初始化相关参数
	修改:
	前两个不用修改,直接放到caffe_root目录下
	对 3)的修改: 
	<1> BATCHSIZE ; BATCHNUM : 要保证BATCHSIZE*BATCHNUM = 图片总数
	<2> 还需要保证 BATCHSIZE * batch_size(位于deploy.prototxt中) >= 图片总数
	另外,如果要提取其他层的特征,需要修改对应的 Layer, Dim,和Out路径
2. 主程序
	python SBD_CNN.py "videoName: anni009.mpg"
	注意修改存放的向量 matfile 和 输出结果 Notes 的名称
