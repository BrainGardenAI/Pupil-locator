import json


class HierarchicalDict:
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
        return self.storage
    
    def __str__(self):
        return str(self.storage)
    
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
