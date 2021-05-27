# -----------------------------------------------------------------------------
'''
&usage:		HMM-GMM的异常心音识别模型
@author:	renyu 
'''
# -----------------------------------------------------------------------------
from python_speech_features import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
from sklearn.externals import joblib
import numpy as np
from scipy import signal
import os
import random
#import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
'''
&usage:		准备所需数据
'''
# -----------------------------------------------------------------------------
# 生成wavdict，key=wavid，value=wavfile
def gen_wavlist(wavpath):
	wavdict = {}
	labeldict = {}
	t1 = 0
	t2 = 0
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		lis = list(range(len(filenames)))
		random.shuffle(lis)
		for i in range(len(lis)):
			j = lis[i]
			filename = filenames[j]
#			print(filename)
#		for filename in filenames:
			if filename.endswith('.wav'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				wavdict[fileid] = filepath
				label = fileid.split('-')[1]
#				labeldict[fileid] = label
				if label == '0':
					t1 = t1 + 1
					labeldict[fileid] = '0'
				else:
					t2 = t2 + 1
					labeldict[fileid] = '1'
	print(t1)
	print(t2)
	return wavdict, labeldict

# 特征提取，feat = compute_mfcc(wadict[wavid])
def compute_mfcc(file):
	fs, audio = wavfile.read(file)
	b, a = signal.butter(6, [50/8000,1000/8000], 'bandpass')  #配置滤波器 8 表示滤波器的阶数
	audio = signal.filtfilt(b, a, audio) #data为要过滤的信号       
	# 这里我故意fs/2,有些类似减小step，不建议这样做，投机取巧做法
	mfcc_feat = mfcc(audio, samplerate= fs, numcep=26, winlen=0.012, nfft=96)   #### winlen=0.01, winstep=0.005可行
	return mfcc_feat
# -----------------------------------------------------------------------------
'''
&usage:		搭建HMM-GMM的异常心音识别模型
参数意义:
	CATEGORY:	所有标签的列表
	n_comp:		每个心音中的状态数
	n_mix:		每个状态包含的混合高斯数量
	cov_type:	协方差矩阵的类型
	n_iter:		训练迭代次数
'''
# -----------------------------------------------------------------------------
class Model():
	"""docstring for Model"""
	def __init__(self, CATEGORY=None, n_comp = 4, n_mix = 4, cov_type='diag', n_iter= 200):
		#super(Model, self).__init__()
		self.CATEGORY = CATEGORY
		self.category = len(CATEGORY)
		self.n_comp = n_comp
		self.n_mix = n_mix
		self.cov_type = cov_type
		self.n_iter = n_iter
		# 关键步骤，初始化models，返回特定参数的模型的列表
		self.models = []
		for k in range(self.category):
			model = hmm.GMMHMM(n_components=self.n_comp, n_mix = self.n_mix, 
								covariance_type=self.cov_type, n_iter=self.n_iter)
			self.models.append(model)

	# 模型训练
	def train(self, wavdict=None, labeldict=None):
		se = []
		for k in range(2):                                     ###############################################要改
			model = self.models[k]
			for x in wavdict:
				if labeldict[x] == self.CATEGORY[k]:
					mfcc_feat = compute_mfcc(wavdict[x])
                    
					model.fit(mfcc_feat)
#			model.last_()                                      ##################################################自己加
#			model.prin()
			mfcc_feat = compute_mfcc(wavdict[x])
#			se.append(model.predict(mfcc_feat))
#画图
#			temp0 = []                                         ##################################################自己加
#			temp1 = []                                         ##################################################自己加
#			for i in range(len(se[k])-1):                      ##################################################自己加
#				if se[k][i]!=se[k][i+1]:                       ##################################################自己加
#					temp0.append(i*40)                         ##################################################自己加
#					temp1.append(0)                            ##################################################自己加
#			plt.plot(np.arange(len(audio)),audio,'k-',temp0,temp1,'r^')######################################自己加
#			tit = 'C:/Users/Administrator/Desktop/save.png'
#			plt.savefig(tit)
#			plt.show()					                       ##################################################自己加
		return se, mfcc_feat #状态序列

	# 使用特定的测试集合进行测试 
	def test(self, wavdict=None, labeldict=None):
		result = []
		for k in range(self.category):         
			subre = []
			label = []
			model = self.models[k]
			for x in wavdict:
				se = []
				mfcc_feat= compute_mfcc(wavdict[x])
				# 生成每个数据在当前模型下的得分情况
				re = model.score(mfcc_feat)
				subre.append(re)
				label.append(labeldict[x])
				se.append(model.predict(mfcc_feat))
#画图
#				temp0 = []                                     ##################################################自己加
#				temp1 =np.zeros(len(audio))
#				da = max(abs(audio))                           ##################################################自己加
#				for i in range(len(se[0])-1):                  ##################################################自己加
#				    if se[0][i]!=se[0][i+1]:                   ##################################################自己加
#					      temp0.append(i*40)                   ##################################################自己加
#					      temp1[i*40]=-da                      ##################################################自己加
#					      temp1[i*40+1]=da                     ##################################################自己加
#				for i in range(700):
#				    temp1[i]=0
#				for i in range(len(audio)-700,len(audio)):
#				    temp1[i]=0
#				plt.plot(np.arange(len(audio)),audio,'k-',np.arange(len(temp1)),temp1,'k-.')    ##################################自己加
#				tit = 'C:/Users/Administrator/Desktop/save.png'
#				plt.savefig(tit)
#				plt.show()	
#				print(se[0])
			# 汇总得分情况
			result.append(subre)
			result_two_problity = result
		# 选取得分最高的种类
		result = np.vstack(result).argmax(axis=0)
		# 返回种类的类别标签
		result = [self.CATEGORY[label] for label in result]
#		print('识别得到结果：\n',result)
#		print('原始标签类别：\n',label)
		# 检查识别率，为：正确识别的个数/总数
		TP = 0 #阳性识别为阳性
		FP = 0 #阴性识别为阳性
		TN = 0 #阴性识别为阴性
		FN = 0 #阳性识别为阴性
        #分多类
		for i in range(len(result)):
			if label[i] == '1':
				if result[i] == '1':
					TP = TP + 1
				else:
					FN = FN + 1
			else:
				if result[i] == '0':
					TN = TN + 1
				else:
					FP = FP + 1
		print('TP: ' + str(TP))
		print('FP: ' + str(FP))
		print('TN: ' + str(TN))
		print('FN: ' + str(FN))
		Accuracy=(TP+TN)/(TP+TN+FP+FN)
		Precision=TP/(TP+FP)
		Sensitivity=TP/(TP+FN)
		Specificity=TN/(FP+TN)
		F1_score=2*Precision*Sensitivity/(Precision+Sensitivity)
		print('Accuracy:' + str(Accuracy))
		print('Precision:' + str(Precision))
		print('Sensitivity:' + str(Sensitivity))
		print('Specificity:' + str(Specificity))
		print('F1_score:' + str(F1_score))
		return result_two_problity
        
	def save(self, path="models.pkl"):
		# 利用external joblib保存生成的hmm模型
		joblib.dump(self.models, path)


	def load(self, path="models.pkl"):
		# 导入hmm模型
		self.models = joblib.load(path)
# -----------------------------------------------------------------------------
'''
&usage:		使用模型进行训练和识别
'''
# -----------------------------------------------------------------------------
# 准备训练所需数据
CATEGORY = ['0','1']######################################################要改
traindict, trainlabel = gen_wavlist('train_data4000')
testdict, testlabel = gen_wavlist('test_data')	
# 进行训练
models = Model(CATEGORY=CATEGORY)
se,mfcc_feat = models.train(wavdict=traindict, labeldict=trainlabel)###不要训练时可注释掉
models.save()###不要训练时可注释掉
models.load()
models.test(wavdict=traindict, labeldict=trainlabel)
result = models.test(wavdict=testdict, labeldict=testlabel)