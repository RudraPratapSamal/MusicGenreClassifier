import sys
import os
import time
import numpy as np
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *		
from qt_material import apply_stylesheet
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd

class mainWindow(QStackedWidget):

	def __init__(self):
		QStackedWidget.__init__(self)

		self.setWindowTitle("Music Genre Classifier")
		self.setFixedSize(400,600)
		self.setWindowIcon(QIcon('thumbnail.png'))

		self.mainWidget=QWidget()
		self.addWidget(self.mainWidget)

		self.screen()

		return

	def screen(self):
		self.setCurrentWidget(self.mainWidget)

		head=QWidget()
		head.setFixedHeight(100)

		mid=QWidget()
		mid.setFixedHeight(350)
		

		foot=QWidget()
		foot.setFixedHeight(130)

		grid=QGridLayout()
		self.mainWidget.setLayout(grid)

		grid.addWidget(head,0,0)
		grid.addWidget(mid,1,0)
		grid.addWidget(foot,2,0)


		#Head
		self.browseButton=QPushButton()
		self.browseButton.setText('Browse')
		self.browseButton.clicked.connect(self.browse_aud)


		self.submitButton=QPushButton()
		self.submitButton.setText('Submit')
		self.submitButton.clicked.connect(self.submit_aud)
		self.submitButton.setEnabled(False)

		self.featuresButton=QPushButton()
		self.featuresButton.setText('Features')
		self.featuresButton.setEnabled(False)
		self.featuresButton.clicked.connect(self.extract_features)

		self.audDest=QLineEdit()
		self.audDest.setEnabled(False)

		filenNameLabel=QLabel()
		filenNameLabel.setText('File Path')

		self.loadingLabel=QLabel()
		self.loadingLabel.setText('Loading')

		self.progBar = QProgressBar()
		self.progBar.setValue(0)
		self.progBar.setEnabled(False)

		self.headGrid=QGridLayout()
		head.setLayout(self.headGrid)

		self.headGrid.addWidget(filenNameLabel,0,0)
		self.headGrid.addWidget(self.audDest,0,1,1,3)
		#self.headGrid.addWidget(self.browseButton,0,3)
		#self.headGrid.addWidget(self.submitButton,0,4)
		self.headGrid.addWidget(self.progBar,1,1,1,3)
		self.headGrid.addWidget(self.loadingLabel,1,0)
		#self.headGrid.addWidget(self.featuresButton,1,3,1,2)
		self.headGrid.addWidget(self.browseButton,2,1)
		self.headGrid.addWidget(self.submitButton,2,2)
		self.headGrid.addWidget(self.featuresButton,2,3)
		#self.headGrid.addWidget(noteLabel,3,0,1,4)

		#Mid
		noteLabel=QLabel()
		noteLabel.setText('*The below listed features are required to classify genre of any music')
		noteLabel.setWordWrap(True)

		self.featuresTable=QTableWidget()
		self.featuresTable.setRowCount(29)
		self.featuresTable.setColumnCount(2)
		self.featuresTable.setColumnWidth(0,174)
		self.featuresTable.setColumnWidth(1,174)
		self.featuresTable.horizontalHeader().setVisible(False)
		self.featuresTable.verticalHeader().setVisible(False)

		font=QFont()
		font.setBold(True)

		self.featuresTable.setItem(0,0, QTableWidgetItem('Features'))
		self.featuresTable.item(0,0).setFont(font)

		self.featuresTable.setItem(0,1, QTableWidgetItem('Values'))
		self.featuresTable.item(0,1).setFont(font)

		features=['chroma_stft_mean','rms_mean','spectral_centroid_mean','spectral_bandwidth_mean','rolloff_mean','zero_crossing_rate_mean','harmony_mean','tempo','mfcc1_mean','mfcc2_mean','mfcc3_mean','mfcc4_mean','mfcc5_mean','mfcc6_mean','mfcc7_mean','mfcc8_mean','mfcc9_mean','mfcc10_mean','mfcc11_mean','mfcc12_mean','mfcc13_mean','mfcc14_mean','mfcc15_mean','mfcc16_mean','mfcc17_mean','mfcc18_mean','mfcc19_mean','mfcc20_mean']
		row=1

		for i in features:
			self.featuresTable.setItem(row,0, QTableWidgetItem(i))
			row+=1

		self.featuresTable.setEditTriggers(QAbstractItemView.NoEditTriggers)

		self.midGrid=QGridLayout()
		mid.setLayout(self.midGrid)

		self.midGrid.addWidget(noteLabel,0,0)
		self.midGrid.addWidget(self.featuresTable,1,0)


		#Foot
		self.resultButton=QPushButton()
		self.resultButton.setText('Show Genre')
		self.resultButton.setEnabled(False)
		self.resultButton.clicked.connect(self.result_genre)

		pic=QPixmap('thumbnail50.png')
		thumbnail=QLabel()
		thumbnail.setPixmap(pic)
		thumbnail.resize(50,50)

		nameLabel=QLabel()
		nameLabel.setText('Name')
		nameLabel.setFont(font)

		genreLabel=QLabel()
		genreLabel.setText('Genre')
		genreLabel.setFont(font)


		self.audioNameLabel=QLabel()
		self.audioGenreLabel=QLabel()

		self.clearButton=QPushButton()
		self.clearButton.setText('Clear All')
		self.clearButton.setEnabled(False)
		self.clearButton.clicked.connect(self.clear_all)

		self.exitButton=QPushButton()
		self.exitButton.setText('Exit')
		self.exitButton.clicked.connect(self.exit_dialog)

		self.footGrid=QGridLayout()
		foot.setLayout(self.footGrid)

		self.footGrid.addWidget(self.resultButton,0,0,1,5)
		self.footGrid.addWidget(thumbnail,1,0,2,1)
		self.footGrid.addWidget(nameLabel,1,1)
		self.footGrid.addWidget(genreLabel,2,1)
		self.footGrid.addWidget(self.audioNameLabel,1,2,1,3)
		self.footGrid.addWidget(self.audioGenreLabel,2,2,1,3)
		self.footGrid.addWidget(self.clearButton,3,0)
		self.footGrid.addWidget(self.exitButton,3,4)

		return


	def browse_aud(self):
		self.filePath , check = QFileDialog.getOpenFileName(None, "Music File","", "Wave Audio File (*.wav)")
		self.audDest.setText(self.filePath)
		if check:
			self.submitButton.setEnabled(True)
			self.browseButton.setEnabled(False)
			self.progBar.setEnabled(True)
		else:
			self.submitButton.setEnabled(False)
			self.browseButton.setEnabled(True)
			self.progBar.setEnabled(False)
		print(os.path.basename(self.filePath))

		return


	def submit_aud(self):
		self.submitButton.setEnabled(False)
		y, sr = librosa.load(self.filePath)
		x=(librosa.get_duration(y=y, sr=sr))/2000
		
		for i in range(101):
			time.sleep(x)
			self.progBar.setValue(i)
		self.featuresButton.setEnabled(True)
		
		
		return

	
	def extract_features(self):
		self.featuresButton.setEnabled(False)
		self.clearButton.setEnabled(True)

		y,sr=librosa.load(self.filePath,duration=30)

		self.chm=np.mean(np.mean(librosa.feature.chroma_stft(y=y, sr=sr),axis=1),axis=0)

		s,phase=librosa.magphase(librosa.stft(y))
		self.rmm=np.mean(np.mean(librosa.feature.rms(S=s),axis=1),axis=0)

		self.scm=np.mean(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr),axis=1),axis=0)

		self.sbm=np.mean(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr),axis=1),axis=0)

		self.rom=np.mean(np.mean(librosa.feature.spectral_rolloff(S=s, sr=sr),axis=1),axis=0)

		self.zcm=np.mean(np.mean(librosa.feature.zero_crossing_rate(y),axis=1),axis=0)


		self.hm=np.mean(np.mean(librosa.effects.harmonic(y)))

		tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr), sr=sr)
		self.tempo=tempo[0]

		mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
		self.mfc_mean={}

		count=0
		for i in mfccs:
			count+=1
			self.mfc_mean.update({count:np.mean(i)})

		self.extractedList=[self.chm,self.rmm,self.scm,self.sbm,self.rom,self.zcm,self.hm,self.tempo,self.mfc_mean[1],self.mfc_mean[2],self.mfc_mean[3],self.mfc_mean[4],self.mfc_mean[5],self.mfc_mean[6],self.mfc_mean[7],self.mfc_mean[8],self.mfc_mean[9],self.mfc_mean[10],self.mfc_mean[11],self.mfc_mean[12],self.mfc_mean[13],self.mfc_mean[14],self.mfc_mean[15],self.mfc_mean[16],self.mfc_mean[17],self.mfc_mean[18],self.mfc_mean[19],self.mfc_mean[20]]

		row=1
		for i in self.extractedList:
			self.featuresTable.setItem(row,1, QTableWidgetItem(str(i)))
			row+=1

		self.compute_genre()

		self.resultButton.setEnabled(True)

		return


	def clear_all(self):
		self.browseButton.setEnabled(True)
		self.submitButton.setEnabled(False)
		self.featuresButton.setEnabled(False)
		self.resultButton.setEnabled(False)
		self.clearButton.setEnabled(False)
		'''for i in reversed(range(self.headGrid.count())):
			self.headGrid.itemAt(i).widget().setParent(None)'''
		a=''
		self.audDest.setText(a)
		self.audioNameLabel.setText(a)
		self.audioGenreLabel.setText(a)
		self.progBar.setValue(0)
		row=1
		for i in self.extractedList:
			self.featuresTable.setItem(row,1, QTableWidgetItem(a))
			row+=1
		return


	def compute_genre(self):
		df=pd.read_csv('Data/features_30_sec.csv')

		xVal=df[['chroma_stft_mean','rms_mean','spectral_centroid_mean','spectral_bandwidth_mean','rolloff_mean','zero_crossing_rate_mean','harmony_mean','tempo','mfcc1_mean','mfcc2_mean','mfcc3_mean','mfcc4_mean','mfcc5_mean','mfcc6_mean','mfcc7_mean','mfcc8_mean','mfcc9_mean','mfcc10_mean','mfcc11_mean','mfcc12_mean','mfcc13_mean','mfcc14_mean','mfcc15_mean','mfcc16_mean','mfcc17_mean','mfcc18_mean','mfcc19_mean','mfcc20_mean']].values
		yVal=df['label'].values

		xTrain,xTest,yTrain,yTest=train_test_split(xVal,yVal,test_size=0.2,random_state=4)

		Ks = 10
		meanAcc = np.zeros((Ks-1))
		predX=[[self.chm,self.rmm,self.scm,self.sbm,self.rom,self.zcm,self.hm,self.tempo,self.mfc_mean[1],self.mfc_mean[2],self.mfc_mean[3],self.mfc_mean[4],self.mfc_mean[5],self.mfc_mean[6],self.mfc_mean[7],self.mfc_mean[8],self.mfc_mean[9],self.mfc_mean[10],self.mfc_mean[11],self.mfc_mean[12],self.mfc_mean[13],self.mfc_mean[14],self.mfc_mean[15],self.mfc_mean[16],self.mfc_mean[17],self.mfc_mean[18],self.mfc_mean[19],self.mfc_mean[20]]]
		for n in range(1,Ks):
		    #Train Model
		    neigh = KNeighborsClassifier(n_neighbors = n).fit(xTrain,yTrain)
		    yhat=neigh.predict(xTest)
		    meanAcc[n-1] = metrics.accuracy_score(yTest, yhat)

		#Prediction
		bestNeigh = KNeighborsClassifier(n_neighbors = 1).fit(xTrain,yTrain)

		self.genre=bestNeigh.predict(predX)
		print(self.genre)
		
		return


	def result_genre(self):
		self.audioNameLabel.setText(os.path.basename(self.filePath))
		self.audioGenreLabel.setText(self.genre[0].capitalize())
		self.resultButton.setEnabled(False)
		
		return


	def exit_dialog(self):
		msgDialog=QMessageBox()
		msgDialog.setIcon(QMessageBox.Question)
		msgDialog.setWindowTitle('Exit Music Genre Classifier')
		msgDialog.setText('Do you want to exit Music Genre Classifier')
		msgDialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		msgDialog.setDefaultButton(QMessageBox.No)
		retValue=msgDialog.exec()
		if retValue==QMessageBox.Yes:
			self.close()
		else:
			pass

		return


app=QApplication(sys.argv)
app.setStyle('Fusion')
apply_stylesheet(app, theme='light_lightgreen.xml', invert_secondary=True)
stylesheet = app.styleSheet()
with open('green.css') as file:
    app.setStyleSheet(stylesheet + file.read().format(**os.environ))
ex=mainWindow()
ex.show()
sys.exit(app.exec_())