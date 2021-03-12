from refer import REFER
import json
import numpy as np

refer = REFER('/home/tanmayg/Data/gpv/learning_phase_data/refcocop/anno', dataset='refcoco+', splitBy='unc')
outdir = '/home/tanmayg/Data/gpv/learning_phase_data/refcocop/'


def write(file_name, data, mode='wb'):
    with open(file_name, mode) as f:
        f.write(data)


def get_split(subset):
    ref_ids = refer.getRefIds(split=subset)
    image_ids = refer.getImgIds(ref_ids)
    refs = refer.loadRefs(ref_ids)
    data = []
    for ref in refs:
        ref_id = ref['ref_id']
        anno = refer.refToAnn[ref_id]
        image_id = ref['image_id']
        image_subset = ref['file_name'].split('_')[1]
        for sent in ref['sentences']:
            query = sent['sent']
            box = anno['bbox']
            sent_id = sent['sent_id']
            sample = {
                'boxes': [box],
                'sent_id': sent_id,
                'image': {
                    'image_id': image_id,
                    'subset': image_subset
                },
                'query': query
            }
            data.append(sample)
    
    print(subset,len(data))
    json_path = outdir + subset + '.json'
    data = json.dumps(data, sort_keys=True, indent=4)
    write(json_path, data, 'w')


def subsample_train(per,seed=0):
    json_path = outdir + 'train.json'
    data = json.load(open(json_path,'r'))
    N = len(data)
    n = int(per*N/100)
    np.random.seed(seed)
    data = np.random.choice(data,n,replace=False).tolist()
    print(per,len(data))
    json_path = outdir + 'train_' + str(per) + '.json'
    data = json.dumps(data, sort_keys=True, indent=4)
    write(json_path, data, 'w')


if __name__=='__main__':
    for subset in ['train','val','test']:
        get_split(subset)
        
    for per in [1,2,5,10,25,50,75,100]:
        subsample_train(per)