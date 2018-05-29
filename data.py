import cv2
import os
import numpy
import numpy as np

Filelist = os.listdir("faces")

dic = "faces"
test = []
for var in Filelist[0:3]:
    path = os.path.join(dic, var)
    test.append(cv2.imread(path).astype(np.float32))
test = np.array(test)
test.shape[0]

class dataset(object):
    def __init__(self, dictionary, max_num = None):
        assert os.path.exists(dictionary)
        
        data = []
        self._num_examples = 0
        self._epoch_completed = 0
        self._index_in_epoch = 0
        
        FileList = os.listdir(dictionary)
        for FileName in FileList:
            data.append(cv2.imread(os.path.join(dictionary, FileNmae)).astype(np.float32))
            self._num_examples+=1
            if self._num_examples == max_num:
                break
        self._image = np.array(data)
        
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_examples:
            self._epoch_completed += 1
            perm = numpy.arange(self._num_examples)
            np.random.shuffle(perm)
            self._image = self._image[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert self._index_in_epoch <= batch_size
            
        end = self._index_in_epoch
        
        return data[start:end]
    
    @property
    def image(self):
        return self._image
    @property
    def epoch(self):
        return self._epoch_completed
    @property
    def num_examples(self):
        return self._num_examples



def input_data(dic_A, dic_B, max_num = None):
    class datasets(object):
        pass
    
    img_data = datasets()
    img_data.A = dataset(dic_A, max_num)
    img_data.B = dataset(dic_B, max_num)
    assert img_data.A.num_examples == img_data.B.num_examples
    img_data.num = img_data.A.num_examples
    
    return img_data

