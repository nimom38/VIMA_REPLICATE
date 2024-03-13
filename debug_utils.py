import numpy as np
import json

def print_to_file(filename, obj):
  def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

  with open(filename, 'w') as file:
    json.dump(obj, file, default=ndarray_to_list, indent=4)