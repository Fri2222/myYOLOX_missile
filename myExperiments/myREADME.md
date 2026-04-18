###################################################################################################################

项目名称:yolox
环境名称:yolox_env

####################################################################################################################


1.用训练完成的权重文件识别视频demo
python tools\track_video.py -p "视频地址"

2.测试前端YOLOX识别图片

python tools/demo.py image -f exps/default/yolox_missile.py -c YOLOX_outputs/yolox_s_missile/best_ckpt.pth --path "图片地址" --conf 0.25 --save_result --device gpu --fp16 


####################################################################################################################
anaconda命令

1.conda info --envs

2.conda activate yolox_env

3.cd H:\Code\YOLOX\

####################################################################################################################
git命令

1. 添加所有文件到暂存区
	git add .

2.提交文件到本地仓库
	git commit -m "这里写备注"
	
3.将本地代码推送到 GitHub
	git push -u origin <分支名>
	
4.查看当前分支
	git branch
	
5.创建分支
	git branch <分支名>
	
6.切换分支
	git checkout <分支名>