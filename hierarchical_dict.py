import json


class HierarchicalDict:
    """
    This class implements dictionary with hierarchical structure. 
    It is convinient when you need to store data in a hierarchical way.
    Suppose you have some data folder with a structure like this:
        Actor1
            Real
                VideoSegment1
                    Frame1
                    Frame2
                    ...
                VideoSegment2
                    Frame1
                    ...
            Virtual
                VideoSegment1
                    Frame1
                    ...
                ...
        Actor2
            ...
    The deeper the hierarchy, the more unique keys you need to use in order to get to the lowest level.
    To make the code more simple one can use this class to assign or retrieve elements from different levels 
    by passing list of keys to  __setitem__ or __getitem__ methods. 
    In such list i-th element represents the key for i-th level in the hierarchy.

    Example:
        instead of writing like this
        frame = Storage[ActorKey][DomainKey][VideoKey][FrameKey]

        you can write like this:
        keys = [ActorKey, DomainKey, VideoKey, FrameKey]
        frame = Storage[keys]
    """

    def __init__(self, path=None):
        self.storage = {}
        if path:
            self.storage = self.load_json(path)
        
    def __setitem__(self, keys, item):
        current_level = self.storage
        for k in keys[:-1]:
            if k not in current_level:
                current_level[k] = {}
            current_level = current_level[k]
        last_key = keys[-1]
        current_level[last_key] = item
    
    def __getitem__(self, keys):
        current_level = self.storage
        for k in keys:
            current_level = current_level[k]
        return current_level
    
    def __repr__(self):
        return str(self.storage)
    
    def __str__(self):
        return str(self.storage)

    def check_key(self, keys):
        current_level = self.storage
        for k in keys:
            if k not in current_level.keys():
                return False
            current_level = current_level[k]
        return True
    
    def save_json(self, path):
        with open(path, 'w') as fp:
            json.dump(self.storage, fp)
    
    def load_json(self, path):
        with open(path, 'r') as fp:
            return json.load(fp)


if __name__ == '__main__':
    HD = HierarchicalDict()
    keys = ['NormanReedus', 'Real', 'Segment_1', '001']
    item = 1
    HD[keys] = item

    keys = ['NormanReedus', 'Real', 'Segment_1', '002']
    item = 2
    HD[keys] = item

    keys = ['NormanReedus', 'Virtual', 'Segment_1', '002']
    item = 2
    HD[keys] = item
    print(HD)

    keys = ['NormanReedus', 'Virtual', 'Segment_1', '002']
    item = 1
    HD[keys] = item

    keys = ['NormanReedus', 'Real', 'Segment_1', '002']
    item = 3
    HD[keys] = item
    print(HD)
