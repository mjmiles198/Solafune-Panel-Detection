import os
if not os.path.isdir('output'):
    os.mkdir('output')


test_path =  '/tmp/kaggledata/solafune_solars/evaluation/'
test_mask_path = '/tmp/kaggledata/solafune_solars/sample/'
#test_path =  '/tmp/kaggledata/solafue_solars/train/s2_image/'
#test_mask_path = '/tmp/kaggledata/solafue_solars/train/mask/'

masks = glob.glob(f'{test_mask_path}/*')
tests = glob.glob(f'{test_path}/*')
masks.sort()
tests.sort()

threshold = 0.5

for i, (m, t) in tqdm.tqdm(enumerate(zip(masks, tests))):
    basename = os.path.basename(m)
    output_file = f'output/{basename}'
    
    img = tifffile.imread(t).astype(np.float)
    mask = tifffile.imread(m).astype(np.float)
    
    X = img.reshape(-1, 12) 
    shape_mask = mask.shape
    
    pred = 0
    for model in models:
        pred = model.predict_proba(X) / len(models)

    pred_mask = np.argmax(pred, axis=1).astype(np.uint8)
    pred_mask = pred_mask.reshape(shape_mask[0], shape_mask[1])
    
    tifffile.imwrite(output_file, pred_mask)
