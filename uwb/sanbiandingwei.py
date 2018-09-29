import serial
import time
import thread
import numpy as np

#ser=serial.Serial("/dev/ttyUSB0",115200)
#ser=serial.Serial("COM3",115200)

#基站坐标基本信息
basestationDicts={"1":np.array([2630.00,2700.00,30.00]), #单位mm
                  "2":np.array([2600.00,3540.00,800.00]),
                  "3":np.array([800.00,4000.00,30.00]),
                  "4":np.array([480.00,3720.00,620.00])}
#保存标签坐标信息的数据文件路径
tagaxisFilePath="tagaxis.txt"                  
                  
#串口多线程读取
class MSerialPort:
    message=''
    tagaxis=[] #列表数据
    def __init__(self,port,buad):
        self.port=serial.Serial(port,buad)
        if not self.port.isOpen():
            self.port.open()
    def port_open(self):
        if not self.port.isOpen():
            self.port.open()
    def port_close(self):
        self.port.close()
        np.savetxt(tagaxisFilePath,self.tagaxis,fmt="%.2f",delimiter=",")
    def send_data(self,data):
        number=self.write(data)
        return number
    ## 解析串口收到的字符串，并计算tag坐标    
    def dealRecvStr(self,recvMessage):
        mapx={}
        strArray=recvMessage.split("\r\n")
        for i in range(len(strArray)):
            if(strArray[i].startswith("distance") and strArray[i].find("mm")>0 and  strArray[i].find(":")>0):
               # print "*****************************************"
               # print strArray[i]
                locationName=strArray[i][(strArray[i].index("distance")+8):(strArray[i].index(":"))].strip()
                distance=strArray[i][(strArray[i].index(":")+1):(strArray[i].index("mm"))].strip()                
                try:
                    mapx[locationName]=np.double(distance)
                except:
                    continue
        if(len(mapx)<4):
            print "can not calculate the tag axis as distance data count is less than 4"
            return
        baseStationInfo=np.zeros((len(mapx),4))
        index=0
        for key in mapx:  
            if(basestationDicts.has_key(key)):
                baseStationInfo[index,0]=basestationDicts[key][0]
                baseStationInfo[index,1]=basestationDicts[key][1]
                baseStationInfo[index,2]=basestationDicts[key][2]
                baseStationInfo[index,3]=mapx[key]
            else:
                return
            index=index+1
        originalX=np.array([0.0,0.0,0.0])
        insNewton=MultivariateNewton(originalX,0.00001,100,baseStationInfo)
        x,y=insNewton.getNewtonMin()
        print "caculate axis:*****************************"
        print x,y
        print "calculate over****************************"
        self.tagaxis.append(x)
        
    def read_data(self):
        cnt=0
        while True:
            data=self.port.readline()
            self.message+=data
            cnt=cnt+1
            if(cnt>30):
                cnt=0
                print("----------------------------------------")
                print self.message
                recvStr=self.message
                self.message=''
                self.dealRecvStr(recvStr)
                

class MultivariateNewton(object):
#    def __init__(self):
#        self.originalX=np.zeros(3,1)
#        self.e=0.0
#        self.maxCycle=100;
#        #基站坐标信息
#        self.baseStationInfo=np.array([2630.00,2700.00,30.00,1279.25],
#                                      [2630.00,3540.00,800.00,1700.5],
#                                      [800.00,4000.00,30.00,2298.5],
#                                      [480.00,3720.00,620.00,1949.00])
                                      
    def __init__(self,originalx,e,maxcycle,baseStationInfo):
        self.originalX=originalx
        self.e=e
        self.maxCycle=maxcycle
        self.baseStationInfo=baseStationInfo
                                      
    def getOriginal(self,x):
        objv=0.0
        for i in range(4):
            sqSum=np.dot(x-self.baseStationInfo[i,0:3],x-self.baseStationInfo[i,0:3])
            fi=np.sqrt(sqSum)-self.baseStationInfo[i,3]
            objv=objv+fi**2
        objv=objv*0.5;
        return objv
        
    def getHessian(self,x):
        #3行乘3列二阶导数矩阵
        jacobian=np.zeros((3,3))
        for i in range(4):
            xxi=x[0]-self.baseStationInfo[i,0]
            xxisq=xxi**2
            yyi=x[1]-self.baseStationInfo[i,1]
            yyisq=yyi**2
            zzi=x[2]-self.baseStationInfo[i,2]
            zzisq=zzi**2
            sumxyzSq=xxisq+yyisq+zzisq
            sumxyzSqCube=sumxyzSq**3
            #二阶导数第一行
            jacobian[0,0]+=1.0-(self.baseStationInfo[i,3]/np.sqrt(sumxyzSq))+(self.baseStationInfo[i,3]*xxisq/np.sqrt(sumxyzSqCube))
            jacobian[0,1]+=self.baseStationInfo[i,3]*xxi*yyi/np.sqrt(sumxyzSqCube)
            jacobian[0,2]+=self.baseStationInfo[i,3]*xxi*zzi/np.sqrt(sumxyzSqCube)
            
            #二阶导数第二行             
            jacobian[1,0]+=self.baseStationInfo[i,3]*xxi*yyi/np.sqrt(sumxyzSqCube)
            jacobian[1,1]+=1.0-(self.baseStationInfo[i,3]/np.sqrt(sumxyzSq))+(self.baseStationInfo[i,3]*yyisq/np.sqrt(sumxyzSqCube))
            jacobian[1,2]+=self.baseStationInfo[i,3]*yyi*zzi/np.sqrt(sumxyzSqCube)
    
            #二阶导数第三行
            jacobian[2,0]+=self.baseStationInfo[i,3]*xxi*zzi/np.sqrt(sumxyzSqCube)           
            jacobian[2,1]+=self.baseStationInfo[i,3]*yyi*zzi/np.sqrt(sumxyzSqCube)  
            jacobian[2,2]+=1.0-(self.baseStationInfo[i,3]/np.sqrt(sumxyzSq))+(self.baseStationInfo[i,3]*zzisq/np.sqrt(sumxyzSqCube))
        return jacobian
        
    def getOneDerivative(self,x):        
        oneDerivative=np.zeros((3,1))
        for i in range(4):
            xxi=x[0]-self.baseStationInfo[i,0]
            xxiSq=xxi**2
            yyi=x[1]-self.baseStationInfo[i,1]
            yyiSq=yyi**2
            zzi=x[2]-self.baseStationInfo[i,2]
            zziSq=zzi**2
            sumxyzSq=xxiSq+yyiSq+zziSq
            fic=(np.sqrt(sumxyzSq)-self.baseStationInfo[i,3])/np.sqrt(sumxyzSq)
            oneDerivative[0,0]+=fic*xxi
            oneDerivative[1,0]+=fic*yyi
            oneDerivative[2,0]+=fic*zzi
        return oneDerivative
            
    def getNewtonMin(self):
        x=self.originalX
        y=0.0
        k=1
        #梯度下降更新公式
        while k<self.maxCycle:
            y=self.getOriginal(x)
            one=self.getOneDerivative(x)
            while(np.abs(one[0,0])<self.e and np.abs(one[1,0])<self.e and np.abs(one[2,0])<self.e):
                break
            two=self.getHessian(x)
            twom=np.mat(two)
            twoInv=twom.I
            twoInv=np.array(twoInv)
           # print twoInv
          #  print one.T
            delta=np.dot(twoInv,one)
            x[0]-=delta[0]
            x[1]-=delta[1]
            x[2]-=delta[2]
            k=k+1
        return x,y       
            
            
        
        
#def main():
#    while True:
#        count=ser.inWaiting()
#        if count!=0:
#            recv=ser.read(count)
#            print recv
#            print("---------------------------------------")
#        ser.flushInput()
#        time.sleep(0.1)

if __name__=='__main__':
    #测试1
#    try:
#        main()
#    except KeyboardInterrupt:
#        ser.close()
    #测试2    
#    originalX=np.array([0.0,0.0,0.0])
#    baseStationInfo=np.array([[2630.00,2700.0,30.0,1279.25],
#                             [2630.0,3540.0,800.0,1700.5],
#                             [800.0,4000.0,30.0,2298.5],
#                             [480.0,3720.0,620.0,1949.5]]
#                             )
#    insNewton=MultivariateNewton(originalX,0.00001,100,baseStationInfo)
#    x,y=insNewton.getNewtonMin()
#    print x,y
    #测试3
    mSerial=MSerialPort('COM3',115200)
    thread.start_new_thread(mSerial.read_data,())
    try:
        while True:
            time.sleep(0.1)
       # print mSerial.message
       # print 'next line'
    except KeyboardInterrupt:
        mSerial.port_close()
