import cv2
import classifier
import imageProcessing
import torch
import updateParkData

import threading
import time

class Controller():
	def __init__(self):
		self.observers = []
		self.parking = Parking()
		self.current = Parking()
		self.thread = None


	def addObserver(self, observer):
		self.observers.append(observer)

	def addSlots(self, slots):
		self.parking.addSlots(slots)
		self.current.addSlots(slots)

	def inform(self):
		for observer in self.observers:
			observer.update(self.parking)

	def start(self):
		cap = cv2.VideoCapture(0) #for camera
		# cap = cv2.VideoCapture('11111.mp4')	#for video
		# model, optimizer, criterion = classifier.load_checkpoint('parking_densenet_4_4.pt')
		model = torch.load('models/squeezenet_t2_v24.pt', map_location = 'cpu')
		model.eval()

		while True:
			ret, frame = cap.read()
			# frame = imageProcessing.adjust_gamma(frame)
			if not ret:
				break
			crop = None
			
			if self.thread != None :
				if not self.thread.is_alive():

					updateThread = threading.Thread(target=checkAndUpdate, args=(self,))
					updateThread.start()

					self.thread = threading.Thread(target=predict, args=[self, frame, model])
					self.thread.start()
			else :
				self.thread = threading.Thread(target=predict, args=[self, frame, model])
				self.thread.start()

			# cv2.imshow('SmartPark', crop)
			for slot in self.current.slots:
				x = slot.x
				y = slot.y
				w = slot.width
				h = slot.height

				if slot.full:
					cv2.rectangle(frame, (x,y),(x+w,y+h),(50, 50, 200),2)
				else:
					cv2.rectangle(frame, (x,y),(x+w,y+h),(153, 228, 239),1)

			if ret:
				cv2.imshow('SmartPark', frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			time.sleep(0.05)

		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()

class ParkingObserver():
	def update(self, parking):

		return

class Parking():
	def __init__(self):
		self.slots = []

	def addSlot(self, slot):
		self.slots.append(slot)

	def addSlots(self, slots):
		for x, y, width, height in slots:
			slotInstance = Slot(x, y, width, height)
			self.addSlot(slotInstance)

	def getSlot(self, id):
		return self.slots[id]

	def isChanged(self, status):
		changed = False
		for i in range(len(slots)):
			if self.slots[i].full != status.slots[i].full:
				# print('isChanged: {}'.format(i))
				changed = True
				self.slots[i].invert()
		return changed


class Slot():
	def __init__(self, x, y, width, height):
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.full = False

	def invert(self):
		self.full = False if self.full else True

	def setStatus(self, status):
		self.full = status

def predict(instance, frame, model):
	for slot in instance.current.slots:
		crop = imageProcessing.extractCrop(frame, slot)
		prediction = classifier.predict(model, crop)
		slot.full = True if prediction else False

def checkAndUpdate(instance):
	if instance.parking.isChanged(instance.current):
				instance.inform()

def constractId(numeric):
	return 'id#{}'.format(numeric);


#=================================================
class Observer(ParkingObserver):
	def update(self, parking):
		print('request')
		response = updateParkData.updateDatabase(parking)
		print(response)
#=================================================

updateParkData.initDatabase() #only for the first comunication

slots = [
			[110, 85, 75, 75],
			[110, 180, 75, 75],
			[520, 87, 75, 75],
			[515, 197, 75, 75],
			[510, 300, 75, 75],
			[505, 400, 75, 75]
		]

controller = Controller()
controller.addSlots(slots)

observer = Observer()
controller.addObserver(observer)
controller.start()