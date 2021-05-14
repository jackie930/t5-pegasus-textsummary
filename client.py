import requests
import numpy as np
import json
# json格式序列调整
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
feature=np.array(range(128))
param = {
	"instances":[
		#每一个大括号是一次请求，里面是每次请求的参数
              {
		"in":feature
	      }
	]
}
param = json.dumps(param, cls=NumpyEncoder)
 
res = requests.post("http://localhost:8501/v1/models/face2:predict", data=param)