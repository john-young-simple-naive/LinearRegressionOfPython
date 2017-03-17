# -*- coding: utf-8 -*-
"""
Encapsulate a Linear Regression Model in a class, input a xlsx file
Created on Tue Mar 07 10:21:54 2017

@author: john107
"""
import matplotlib.pyplot as plt #画图
import numpy as np  #数组格式
from sklearn import linear_model #线性回归模型
import xlrd #读取xlsx文件
from scipy import stats #随机分布
import sklearn


class LR(object):
    def __init__(self,strFilename,strShtname): 
        '''初始化，指定目标xlsx文件和工作表名称'''
        try:
            self.__excel=xlrd.open_workbook(strFilename) #读入文件
            self.__sht=self.__excel.sheet_by_name(strShtname)
        except:
            print '''oops, cann't load the data'''
    def fXindex(self,intXindex):
        '''指定因变量X的列号,必须用中括号括起来'''
        self.__Xindex=intXindex
        self.Xname=list()
        for i in intXindex:
            if 'X' not in dir(self):
                self.X=np.array(self.__sht.col_values(i,start_rowx=1))
            else:
                self.X=np.vstack((self.X,np.array(self.__sht.col_values(i,start_rowx=1))))
            self.Xname.append(self.__sht.cell(0,i).value)
        self.X=self.X.T
    
    def fYindex(self,intYindex): 
        '''指定因变量Y的列号'''
        self.__Yindex=intYindex
        self.Yname=self.__sht.cell(0,intYindex).value
        self.Y=np.array(self.__sht.col_values(intYindex,start_rowx=1))
        
    def fit(self):
        '''回归计算'''
        self.__regr=linear_model.LinearRegression()
        self.__regr.fit(self.X,self.Y)
        self.predictedY=self.__regr.predict(self.X)
        self.residuals=self.Y-self.predictedY
        self.__d1=len(self.__Xindex)
        self.__d2=(self.__sht.nrows-1)-self.__d1-1
        self.__RSS=np.sum(self.residuals**2)
        self.__TSS=np.sum((self.Y-np.mean(self.Y))**2)
        self.__ESS=self.__TSS-self.__RSS
    def __residualsPlot(self):
        '''画残差分布图和残差QQ图'''
        plt.figure(1) #残差分布图
        plt.scatter(self.predictedY,self.residuals)
        plt.xlim((min(self.predictedY),max(self.predictedY)))
        plt.ylim((min(self.residuals),max(self.residuals)))
        plt.title('Residuals VS PredictedY')
        
        plt.figure(2) #残差QQ图
        stdRe=sklearn.preprocessing.scale(self.Y-self.predictedY) #标准化残差
        stats.probplot(stdRe,dist='norm',plot=plt)
        plt.title('Standardrised Residuals QQplot of Norm')
        
        plt.show()
    def __coef(self):
        '''输出阶矩和系数'''
        print 'intercept is: %f'%(self.__regr.intercept_)
        print 'coefficients are: ',
        print self.__regr.coef_
    def __R2(self):
        '''输出拟合优度'''
        print 'R-squared is: %f'%(1-self.__RSS/self.__TSS)
        print 'Adjusted R-squared is: %f'%(1-self.__RSS/self.__TSS*(self.X.shape[0]-1)/(self.X.shape[0]-len(self.__Xindex)-1))
    def __fTest(self):
        '''输出F检验的结果'''
        fValue=(self.__ESS/self.__d1)/(self.__RSS/self.__d2)
        print 'F-value is: %f'%(fValue)
        print 'F-Pvalue is: %f'%(stats.f.pdf(fValue,self.__d1,self.__d2))
    def __tTest(self):
        '''输出t检验的结果'''
        tmpx=np.hstack((np.ones((self.X.shape[0],1)),self.X))
        xTxR=np.linalg.inv(np.dot(tmpx.T,tmpx))
        i=0
        tValue=self.__regr.intercept_/pow(xTxR[i,i]*self.__RSS/self.__d2,1.0/2)
        print 't test results are:'
        print u'----%-15s t-value is: %-10f'%('Intercept',tValue)+u', t-Pvalue is %-10f'%(2*stats.t.cdf(float(-abs(tValue)),self.__d2))
        while i<len(self.__Xindex):
            tValue=self.__regr.coef_[i]/pow(xTxR[i+1,i+1]*self.__RSS/self.__d2,1.0/2)
            print u'----%-15s t-value is %-10f'%(self.Xname[i],tValue)+u', t-Pvalue is %-10f'%(2*stats.t.cdf(float(-abs(tValue)),self.__d2))
            i+=1
    def __vifTest(self):
        '''输出方差膨胀因子检验结果'''
        regr=linear_model.LinearRegression()
        print 'VIF are:'
        #对所有自变量逐个做回归
        for i in range(len(self.__Xindex)):
            tmpIndexx=list()
            tmpIndexy=list()
            for j in range(len(self.__Xindex)):
                tmpIndexx.append(j!=i)
                tmpIndexy.append(j==i)
            tmpIndexx=np.array(tmpIndexx)
            tmpIndexy=np.array(tmpIndexy)
            tx=self.X[:,tmpIndexx]
            ty=self.X[:,tmpIndexy]
            tlm=regr.fit(tx,ty)
            tRSS=np.sum((tlm.predict(tx)-ty)**2)
            tTSS=np.sum((ty-np.mean(ty))**2)
            print '----%-10s: %-8f'%(self.Xname[i],tTSS/tRSS)
    def summary(self):
        self.__residualsPlot()
        self.__coef()
        self.__R2()
        self.__fTest()
        self.__tTest()
        self.__vifTest()
            
if __name__=='__main__':
    exmpLR=LR(unicode(r"F:\\git\\LinearRgr\\mtcars.xlsx","utf_8"),"Sheet 1")
    exmpLR.fXindex([1,2,3,4,5,6,7,8,9,10])
    exmpLR.fYindex(0)
    exmpLR.fit()
    exmpLR.summary()
        
